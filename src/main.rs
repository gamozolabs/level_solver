// cargo add macroquad nalgebra rand

#![feature(array_windows, strict_overflow_ops)]

mod color;

use std::io::Read;
use std::fs::File;
use std::f64::consts::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::collections::BTreeSet;
use ::rand::prelude::*;
use macroquad::prelude::*;

/// Number of surfaces to create (also number of threads)
const SURFACES: usize = 3;

/// Extra Z-scale multiplication for rendering only. This is to visually
/// amplify the Z axis.
const Z_SCALE: f64 = 2000.;

#[derive(Clone)]
struct Samples(
    /// Random sample points and the measurements that contain the points
    /// (xy, dependencies, (sum, count))
    Vec<(DVec2, Vec<usize>, (f64, f64))>,

    /// Lookup from a measurement to samples contained in that measurement.
    /// Indexed by measurement ID
    Vec<Vec<usize>>,

    /// Z adjustment
    f64,
);

impl Samples {
    /// Generate set of samples for `measurements` with `step_size` as the grid
    /// size from the bounding box of the measurements
    fn generate(measurements: &[Measurement], step_size: f64) -> Self {
        let mut samples = Vec::new();

        // Find bounding box of the data, this is the bounding box of all
        // vertices of all triangles of all measurements
        let (bl, tr) = measurements.iter().flat_map(|meas| {
            meas.contact.iter().flat_map(|x| x.array())
        }).bounding_box();

        // Generate samples in a grid
        let mut y = bl.y;
        while y <= tr.y {
            let mut x = bl.x;
            while x <= tr.x {
                samples.push((dvec2(x, y), Vec::new(), (0., 0.)));
                x += step_size;
            }

            y += step_size;
        }

        // Determine which measurements contain the sample points
        for (point, containing, _) in samples.iter_mut() {
            for (ii, meas) in measurements.iter().enumerate() {
                // First check if the point is in any of the bounding boxes
                // for the measurement (rough but cheap)
                let mut found = false;
                for bb in &meas.bounding_boxes {
                    if point.x >= bb.0.x && point.x <= bb.1.x &&
                            point.y >= bb.0.y && point.y <= bb.1.y {
                        found = true;
                        break;
                    }
                }

                if found {
                    // Next check if there is an actual contact triangle that
                    // contains this point
                    for tri in &meas.contact {
                        if tri.contains(*point) {
                            containing.push(ii);
                            break;
                        }
                    }
                }
            }
        }

        // Generate measurements to samples data
        let mut meas_to_samples = vec![Vec::new(); measurements.len()];
        for (samp_id, (_, containing, _)) in samples.iter().enumerate() {
            for &meas_id in containing {
                meas_to_samples[meas_id].push(samp_id);
            }
        }

        Self(samples, meas_to_samples, 0.)
    }

    /// Recompute samples for a given measurement set
    fn recompute(&mut self, measurements: &[Measurement]) {
        // Reset sample points
        self.0.iter_mut().for_each(|(_, _, (sum, count))| {
            *sum = 0.;
            *count = 0.;
        });

        // Sample all points, looking for the minimum value for a plane that
        // contains the point
        for (point, containing, (sum, count)) in self.0.iter_mut() {
            for meas_id in containing {
                *sum   += measurements[*meas_id].plane.get(*point).z;
                *count += 1.;
            }
        }
    }

    /// Normalizes all data to zero as the lowest point
    fn simplify(&mut self) {
        // Find the lowest value
        let mut min = f64::MAX;
        for (_, _, (sum, n)) in self.0.iter_mut() {
            if *n > 0. {
                let val = *sum / *n;
                min = min.min(val);
            }
        }

        // Save adjustment
        self.2 = -min;

        // Adjust all data to zero
        for (_, _, (sum, n)) in self.0.iter_mut() {
            if *n > 0. {
                *sum += self.2 * *n;
            }
        }
    }

    /// Draw the sample points
    fn draw(&mut self, renderer: &mut Renderer, visited: &BTreeSet<usize>,
            min_z: f64, max_z: f64) -> (f64, f64) {
        let range_z = max_z - min_z;

        // Display the data as squares at each point
        for &(loc, ref deps, (s, n)) in self.0.iter() {
            if n > 0. {
                let z = s / n;
                let pct = (z - min_z) / range_z;
                let col = if visited.contains(&deps[0]) {
                    Renderer::color(pct)
                } else {
                    Color::from_rgba(20, 20, 20, 255)
                };
                renderer.draw_square(loc.extend(z), 1.5, col);
            }
        }

        // Flush any pending triangles
        renderer.flush();

        (min_z, max_z)
    }
}

/// A renderer which can cache mesh allocations
struct Renderer {
    /// Internal mesh used during rendering. Used just to prevent reallocs of
    /// backing buffers
    mesh: Mesh,
}

impl Default for Renderer {
    fn default() -> Self {
        Self {
            mesh: Mesh {
                vertices: Vec::new(),
                indices:  Vec::new(),
                texture:  None,
            },
        }
    }
}

impl Renderer {
    /// Render the internal cached mesh
    ///
    /// This assumes the internal mesh is just a list of CCW triangles.
    /// Internally we'll generate `[0..num_verts]` for indices, and compute
    /// normals for the shading of the triangles.
    #[allow(dead_code)]
    fn draw_int(&mut self) {
        // Convenience bindings
        let vertices = &mut self.mesh.vertices;
        let indices  = &mut self.mesh.indices;

        // Generate indices
        indices.resize(vertices.len(), 0);
        indices.iter_mut().enumerate().for_each(|(ii, x)| *x = ii as u16);

        // Render the mesh!
        draw_mesh(&self.mesh);
    }

    /// Flush any triangles pending in the internal mesh
    fn flush(&mut self) {
        if self.mesh.vertices.len() > 0 {
            self.draw_int();
            self.mesh.vertices.clear();
        }
    }

    /// Get the color for a given normalized value [0.0, 1.0]
    fn color(val: f64) -> Color {
        assert!(val >= 0. && val <= 1.0);

        // Inferno color palette lookup
        let idx = (val * 254.99) as usize;
        let partial = (val * 254.99) - idx as f64;
        let col1 = DVec3::from_array(color::INFERNO[idx + 0]);
        let col2 = DVec3::from_array(color::INFERNO[idx + 1]);
        let col = col1 * (1. - partial) + col2 * partial;
        Color::new(col[0] as f32, col[1] as f32, col[2] as f32, 1.)
    }

    // Draw a centered square with a given size and color
    fn draw_square(&mut self, center: DVec3, size: f64, color: Color) {
        // Create two triangles and draw them as a square
        let bl = center + dvec3(-size / 2., -size / 2., 0.);
        let br = center + dvec3( size / 2., -size / 2., 0.);
        let tl = center + dvec3(-size / 2.,  size / 2., 0.);
        let tr = center + dvec3( size / 2.,  size / 2., 0.);

        // Generate both CCW triangles
        for vert in [bl, tr, tl, bl, br, tr] {
            self.mesh.vertices.push(Vertex::new(
                vert.x as f32, vert.y as f32, (vert.z * Z_SCALE) as f32,
                0., 0., color));
        }

        if self.mesh.vertices.len() >= 4096 {
            self.flush();
        }
    }
}

/// A 2d triangle ABC
#[derive(Clone, Copy)]
pub struct Triangle {
    pub a: DVec2,
    pub b: DVec2,
    pub c: DVec2,
}

impl Triangle {
    /// Get the triangle ABC in an array representation
    pub fn array(&self) -> [DVec2; 3] {
        [self.a, self.b, self.c]
    }

    /// Check if the triangle contains a given point
    pub fn contains(&self, point: DVec2) -> bool {
        fn sign(p1: DVec2, p2: DVec2, p3: DVec2) -> f64 {
            (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        }

        let d1 = sign(point, self.a, self.b);
        let d2 = sign(point, self.b, self.c);
        let d3 = sign(point, self.c, self.a);

        let has_neg = (d1 < 0.) || (d2 < 0.) || (d3 < 0.);
        let has_pos = (d1 > 0.) || (d2 > 0.) || (d3 > 0.);

        !(has_neg && has_pos)
    }

    /// Randomly sample exactly `points` from the triangle
    pub fn sample(&self, points: usize) -> Vec<DVec2> {
        // Get the RNG
        let mut rng = ::rand::thread_rng();

        // Sample the points
        let mut samples = Vec::new();
        for _ in 0..points {
            // Generate the two random floats
            let r1: f64 = rng.gen_range(0.0..=1.0);
            let r2: f64 = rng.gen_range(0.0..=1.0);

            // Compute a random point in the triangle
            let point = (1. - r1.sqrt()) * self.a +
                (r1.sqrt() * (1. - r2)) * self.b +
                (r2 * r1.sqrt()) * self.c;

            // Double check our logic
            assert!(self.contains(point));

            // Record the point
            samples.push(point);
        }

        samples
    }

    /// Create a rectangle out of triangles
    pub fn rectangle(bottom_left: DVec2, top_right: DVec2) -> Vec<Self> {
        // Compute all 4 corners of the rectangle
        let bl = bottom_left;
        let br = dvec2(top_right.x, bottom_left.y);
        let tl = dvec2(bottom_left.x, top_right.y);
        let tr = top_right;

        vec![
            Self { a: bl, b: tr, c: tl },
            Self { a: bl, b: br, c: tr },
        ]
    }

    /// Create a polygon out of triangles with a given amount of sides
    pub fn polygon(center: DVec2, radius: f64, sides: usize) -> Vec<Self> {
        // Compute the angle of a side
        let step = TAU / sides as f64;

        let mut tris = Vec::new();
        for side in 0..sides {
            let a1 = DVec2::from_angle((side + 0) as f64 * step);
            let a2 = DVec2::from_angle((side + 1) as f64 * step);

            tris.push(Self {
                a: center,
                b: center + a1.rotate(dvec2(radius, 0.)),
                c: center + a2.rotate(dvec2(radius, 0.)),
            });
        }

        tris
    }

    /// Create a polygon out of triangles with a given amount of sides and
    /// a hole in the center
    pub fn donut(center: DVec2, inner: f64, outer: f64, sides: usize)
            -> Vec<Self> {
        // Compute the angle of a side
        let step = TAU / sides as f64;

        let mut tris = Vec::new();
        for side in 0..sides {
            let a1 = DVec2::from_angle((side + 0) as f64 * step);
            let a2 = DVec2::from_angle((side + 1) as f64 * step);

            tris.push(Self {
                a: center + a1.rotate(dvec2(inner, 0.)),
                b: center + a1.rotate(dvec2(outer, 0.)),
                c: center + a2.rotate(dvec2(outer, 0.)),
            });
            tris.push(Self {
                a: center + a1.rotate(dvec2(inner, 0.)),
                b: center + a2.rotate(dvec2(outer, 0.)),
                c: center + a2.rotate(dvec2(inner, 0.)),
            });
        }

        tris
    }
}

/// An infinite plane which can be sampled for `z` values from a given `xy`
#[derive(Debug, Clone, Copy)]
struct Plane {
    /// Normal of the plane. Must also be normalized for `offset` to have
    /// the correct effect
    normal: DVec3,

    /// Z-offset for the plane
    offset: f64,
}

impl Default for Plane {
    fn default() -> Self {
        Self {
            normal: dvec3(0., 0., 1.),
            offset: 0.,
        }
    }
}

impl Plane {
    /// Get the point on a plane at a given `xy`
    fn get(&self, xy: DVec2) -> DVec3 {
        // Compute Z value for the point
        let z = -(self.normal.x * xy.x + self.normal.y * xy.y - self.offset) /
            self.normal.z;

        DVec3::new(xy.x, xy.y, z)
    }
}

/// Different bounds a value can have
#[derive(Clone, Copy, Debug)]
pub enum Bounds {
    /// An exact value
    Constant(f64),

    /// A uniformly sampled inclusive range [min, max]
    Range(f64, f64),
}

impl Bounds {
    /// Randomly sample the bounds for a value
    fn sample(&self) -> f64 {
        let mut rng = ::rand::thread_rng();

        match *self {
            Self::Constant(val)   => val,
            Self::Range(min, max) => rng.gen_range(min..=max),
        }
    }
}

/// A measurement from the real world
#[derive(Clone)]
struct Measurement {
    /// Contact points for the measurement
    contact: Vec<Triangle>,

    /// Bounding boxes for the contact points of the measurement. This helps
    /// allow for quickly checking if a measurement contains a point, before
    /// checking triangles.
    bounding_boxes: Vec<(DVec2, DVec2)>,

    /// Bounding boxes of the level feet. This is without bounding it to the
    /// surface being measured. This allows us to display where the contact
    /// surfaces of the level are, even if they may be off the surface being
    /// measured
    raw_bounding_boxes: Vec<(DVec2, DVec2)>,

    /// Centroid for the contact patch. This is considered the origin for the
    /// `plane`
    centroid: Option<DVec2>,

    /// The actual plane we attach the triangle to (determines the Z coord for
    /// samples)
    plane: Plane,

    /// X angle (mm/m) of the measurement
    angle_x: Bounds,

    /// Y angle (mm/m) of the measurement
    angle_y: Bounds,

    /// Offset of the backing plane
    offset: Bounds,

    /// Last values picked for the angles and offset
    last_vals: (f64, f64, f64),
}

trait BoundingBox {
    fn bounding_box(self) -> (DVec2, DVec2);
}

impl<T: Iterator<Item = DVec2>> BoundingBox for T {
    fn bounding_box(self) -> (DVec2, DVec2) {
        let mut xmin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;

        for vertex in self {
            xmin = xmin.min(vertex.x);
            ymin = ymin.min(vertex.y);
            xmax = xmax.max(vertex.x);
            ymax = ymax.max(vertex.y);
        }

        (dvec2(xmin, ymin), dvec2(xmax, ymax))
    }
}

impl Measurement {
    /// Compute and/or fetch the centroid for the contact points of the
    /// measurement
    fn centroid(&mut self) -> DVec2 {
        // Get the centroid
        if let Some(centroid) = self.centroid {
            centroid
        } else {
            // Compute the centroid
            let mut centroid = dvec2(0., 0.);
            for tri in &self.contact {
                centroid += tri.a;
                centroid += tri.b;
                centroid += tri.c;
            }
            centroid /= (self.contact.len() * 3) as f64;
            self.centroid = Some(centroid);
            centroid
        }
    }

    /// Recompute the plane based on the X and Y angles
    fn recompute(&mut self, xskew: f64, yskew: f64) {
        // Get the last computed values
        let (angle_x, angle_y, offset) = self.last_vals;
        let angle_x = angle_x + xskew;
        let angle_y = angle_y + yskew;

        // Compute the centroid
        let centroid = self.centroid();

        // Compute plane
        let p = centroid.extend(offset);
        let q = p + dvec3(1000., 0., angle_x);
        let r = p + dvec3(0., 1000., angle_y);

        // Solve for the coefficients of the plane
        self.plane.normal = (q - p).cross(r - p);
        self.plane.offset = self.plane.normal.dot(p);

        #[cfg(debug_assertions)]
        {
            // Validate that our computed plane produces the desired angles
            let x_reading = self.plane.get(dvec2(1000., 0.)).z -
                self.plane.get(dvec2(0., 0.)).z;
            let y_reading = self.plane.get(dvec2(0., 1000.)).z -
                self.plane.get(dvec2(0., 0.)).z;
            let intercept = self.plane.get(centroid);
            assert!(intercept.abs().z < 0.0000001);
            assert!((angle_x - x_reading).abs() < 0.0000001);
            assert!((angle_y - y_reading).abs() < 0.0000001);
        }
    }

    /// Randomly sample the internal bounds and regenerate the plane
    fn mutate(&mut self) {
        // Pick random values for the measurements
        let mut angle_x = self.angle_x.sample();
        let mut angle_y = self.angle_y.sample();
        let mut offset  = self.offset.sample();

        // Move towards the sample
        let bias = 0.001;
        for (old, target) in [
            (self.last_vals.0, &mut angle_x),
            (self.last_vals.1, &mut angle_y),
            (self.last_vals.2, &mut offset),
        ] {
            if !old.is_nan() {
                let val = old * (1. - bias) + *target * bias;
                *target = val;
            } else {
                // Take random initial sample. This is mandatory as the angles
                // may be fixed and thus we must start with them (rather than
                // zero or something).
            }
        }

        // Update the last vals
        self.last_vals = (angle_x, angle_y, offset);
        self.recompute(0., 0.);
    }
}

fn check_angle_mask(val: u32, mask: u32) -> bool {
    (val & 0xf000000) == mask
}

fn integer_to_angle(val: u32) -> f64 {
    const ANGLE_MASK_LOW_BATTERY: u32 = 0x4000000;
    const ANGLE_MASK_LOW_POWER: u32 = 0xc000000;
    const ANGLE_MASK_NEGATIVE_OVERRANGE: u32 = 0x9000000;
    const ANGLE_MASK_POSITIVE_OVERRANGE: u32 = 0x7000000;

    if check_angle_mask(val, ANGLE_MASK_POSITIVE_OVERRANGE) {
        return f64::MAX;
    }

    if check_angle_mask(val, ANGLE_MASK_NEGATIVE_OVERRANGE) {
        return f64::MIN;
    }

    if check_angle_mask(val, ANGLE_MASK_LOW_BATTERY) || check_angle_mask(val, ANGLE_MASK_LOW_POWER) {
        return f64::NAN;
    }

    return (((val & 0xffffffe) << 4) as i32) as f64 / 0x10000000 as f64
}

fn lathe_bed(measurements: &mut Vec<Measurement>) {
    // New casting
    //
    //       250mm center to center
    //       v
    // +--+          +--+
    // |o |==========|  | < 49.4mm square feet
    // +--+          +--+ >>> +x >>>
    //  ||
    //  || < 190mm center to center
    //  ||
    // +--+ v
    // |  | v -y
    // +--+ v
    //
    // Lathe layout
    //
    // +y
    // =======================  < 1240mm
    //   ^
    //   |
    //   v 190mm
    // =================        < 1005mm
    // (0, 0)             +x
    //
    //                 ^
    //                 ^ measurements are relative to 0,0 for each pad
    //                 ^ eg. the bottom left of the pad
    //                 ^
    //                 ^ measurements start -25mm from the end of the short
    //                   way (x = 980mm). Step size is 25mm
    //                   measurements end @ x = -19.4mm

    let foot_square = 49.4;

    // Parse the CSV data in order
    let mut raw_data = Vec::new();
    for filename in [
        //"data/lathe_20241227_afterbend/readings.log",
        //"data/lathe_20241228_scrapetest/level_readings.log",
        //"data/lathe_20250102_stream_readings/level_readings.log",
        //"data/lathe_20250102_stream_readings2/level_readings.log",
        "data/lathe_20250814_custom_casting/level_readings.log",
    ] {
        let mut df = File::open(filename).unwrap();
        let mut chunk = [0u8; 17];
        while df.read_exact(&mut chunk).is_ok() {
            let x_raw = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let y_raw = u32::from_le_bytes(chunk[12..16].try_into().unwrap());
            let trigger = chunk[16] != 0;

            let x_raw = integer_to_angle(x_raw) * 1e3;
            let y_raw = integer_to_angle(y_raw) * 1e3;

            let x_reading =  y_raw;
            let y_reading = -x_raw;

            if trigger {
                println!("M{:03} | Raw X {x_raw:8.3} mm/m | \
                    Raw Y {y_raw:8.3} mm/m",
                    raw_data.len() + 1);
                raw_data.push((x_reading, y_reading));
            }
        }

        println!("Total measurements {}", raw_data.len());
    }

    // Compute coords for data points
    for (ii, &(angle_x, angle_y)) in raw_data.iter().enumerate() {
        // Get the center of the origin foot
        let origin = dvec2(1005. + foot_square / 2., 190.) -
            dvec2(25. * (ii+1) as f64, 0.);

        // Get the center coords of the X and Y feet
        let x_coord = origin + dvec2(250., 0.);
        let y_coord = origin - dvec2(0., 190.);

        // Generate ranges for the X and Y slopes
        let angle_x = Bounds::Constant(angle_x);
        let angle_y = Bounds::Constant(angle_y);

        let mut tris = Vec::new();
        let mut bbs = Vec::new();
        let mut raw_bbs = Vec::new();

        // Generate squares from the center coords of each pad
        for (_ii, &coord) in [origin, x_coord, y_coord].iter().enumerate() {
            let mut c1 = 
                dvec2(coord.x - foot_square / 2., coord.y - foot_square / 2.);
            let mut c2 =
                dvec2(coord.x + foot_square / 2., coord.y + foot_square / 2.);

            // Record the raw bounding box of the shape
            let raw_shape = Triangle::rectangle(c1, c2);
            let raw_bb = raw_shape.iter()
                .flat_map(|x| x.array()).bounding_box();
            raw_bbs.push(raw_bb);

            // Clamp the level contact points to the surface being measured
            if _ii == 0 || _ii == 1 {
                // Origin and X pad are on the long way
                c1.x = c1.x.clamp(0., 1240.);
                c1.y = c1.y.clamp(-12.7 + 190.5, 12.7 + 190.5);
                c2.x = c2.x.clamp(0., 1240.);
                c2.y = c2.y.clamp(-12.7 + 190.5, 12.7 + 190.5);
            } else {
                // Y pad is on the short way
                c1.x = c1.x.clamp(0., 1005.);
                c1.y = c1.y.clamp(-12.7, 12.7);
                c2.x = c2.x.clamp(0., 1005.);
                c2.y = c2.y.clamp(-12.7, 12.7);
            }

            let shape = Triangle::rectangle(c1, c2);

            // Record the bounding box of the shape
            let bb = shape.iter().flat_map(|x| x.array()).bounding_box();
            bbs.push(bb);

            // Record the individual triangles that make up the shape
            tris.extend(shape);
        }

        measurements.push(Measurement {
            contact:  tris,
            centroid: None,
            plane:    Plane::default(),
            offset:   Bounds::Range(-0.25, 0.25),
            last_vals: (f64::NAN, f64::NAN, f64::NAN),
            bounding_boxes: bbs,
            raw_bounding_boxes: raw_bbs,
            angle_x, angle_y,
        });
    }
}

/*
fn lathe_bed_old(measurements: &mut Vec<Measurement>) {
    // Measuring tool dimensions
    //
    //  62.01mm
    //  v
    // +--+      +--+
    // |  |======|  |< 62.20mm
    // +--+      +--+
    //
    // ^------------^
    //  273.76mm

    let across_data = [-0.5,-0.5,0.,-0.5,-1.,0.,-1.,-0.5,0.,0.5,0.,0.,1.,1.,1.,1.,1.5,1.5,1.5,1.5,2.,2.5,2.,2.5,];
    let slopes_long = [1.,0.,-0.5,-0.5,-1.,-1.,-1.,-1.,-0.5,-0.5,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-2.,-2.,-2.,-2.,-2.,-1.5,-2.,-2.,-2.,];
    let slopes_short = [0.,0.,0.,-0.5,0.,0.,0.,0.,0.,-0.5,0.,-0.5,0.,-0.5,-0.5,0.,-0.5,0.,-0.5,-1.,];

    for ii in 1..=across_data.len() {
        // Across is:
        //
        // +--+
        // |  |
        // +--+
        //  ||
        //  ||
        // +--+
        // |  |
        // +--+
        // ^  ^x2
        // ^x1
        let x2 = ii as f64 * 40.;
        let x1 = x2 - 62.20;

        let origin = dvec2(x1, 0.);
        let other  = dvec2(x1, 215.);

        let mut tris = Vec::new();
        let mut bbs = Vec::new();
        for (ii, &coord) in [origin, other].iter().enumerate() {
            let mut coord1 = coord;
            let mut coord2 = coord + dvec2(62.20, 62.01);

            if ii == 0 {
                coord1.x = coord1.x.clamp(0., 1005.);
                coord1.y = coord1.y.clamp(0., 25.4);
                coord2.x = coord2.x.clamp(0., 1005.);
                coord2.y = coord2.y.clamp(0., 25.4);
            } else {
                coord1.x = coord1.x.clamp(0., 1240.);
                coord1.y = coord1.y.clamp(215., 215. + 25.4);
                coord2.x = coord2.x.clamp(0., 1240.);
                coord2.y = coord2.y.clamp(215., 215. + 25.4);
            }

            let shape = Triangle::rectangle(coord1, coord2);

            // Record the bounding box of the shape
            let bb = shape.iter().flat_map(|x| x.array()).bounding_box();
            bbs.push(bb);

            // Record the individual triangles that make up the shape
            tris.extend(shape);
        }

        measurements.push(Measurement {
            contact:  tris,
            centroid: None,
            plane:    Plane::default(),
            offset:   Bounds::Range(-0.25, 0.25),
            last_vals: (f64::NAN, f64::NAN, f64::NAN),
            bounding_boxes: bbs,
            angle_x: Bounds::Range(-0.5, 0.5),
            angle_y: Bounds::Constant(-across_data[ii - 1] * 0.050),
        });
    }

    for ii in 1..=slopes_short.len() {
        // Short way size is 1005mm
        //
        // +--+      +--+
        // |  |======|  | ----> +x
        // +--+      +--+
        // ^  ^x2    ^  ^x4
        // ^x1       ^x3
        let x2 = ii as f64 * 40.;
        let x1 = x2 - 62.01;
        let x4 = x1 + 273.76;
        let x3 = x4 - 62.01;

        let origin = dvec2(x1, 0.);
        let other  = dvec2(x3, 0.);

        let mut tris = Vec::new();
        let mut bbs = Vec::new();
        for (_ii, &coord) in [origin, other].iter().enumerate() {
            let mut coord1 = coord;
            let mut coord2 = coord + dvec2(62.01, 62.20);

            coord1.x = coord1.x.clamp(0., 1005.);
            coord1.y = coord1.y.clamp(0., 25.4);
            coord2.x = coord2.x.clamp(0., 1005.);
            coord2.y = coord2.y.clamp(0., 25.4);

            let shape = Triangle::rectangle(coord1, coord2);

            // Record the bounding box of the shape
            let bb = shape.iter().flat_map(|x| x.array()).bounding_box();
            bbs.push(bb);

            // Record the individual triangles that make up the shape
            tris.extend(shape);
        }

        measurements.push(Measurement {
            contact:  tris,
            centroid: None,
            plane:    Plane::default(),
            offset:   Bounds::Range(-0.25, 0.25),
            last_vals: (f64::NAN, f64::NAN, f64::NAN),
            bounding_boxes: bbs,
            angle_x: Bounds::Constant(slopes_short[ii - 1] * 0.050),
            angle_y: Bounds::Range(-0.5, 0.5),
        });
    }

    for ii in 1..=slopes_long.len() {
        // Long way size is 1240mm
        //
        // +--+      +--+
        // |  |======|  | ----> +x
        // +--+      +--+
        // ^  ^x2    ^  ^x4
        // ^x1       ^x3
        let x2 = ii as f64 * 40.;
        let x1 = x2 - 62.01;
        let x4 = x1 + 273.76;
        let x3 = x4 - 62.01;

        let origin = dvec2(x1, 215.);
        let other  = dvec2(x3, 215.);

        let mut tris = Vec::new();
        let mut bbs = Vec::new();
        for (_ii, &coord) in [origin, other].iter().enumerate() {
            let mut coord1 = coord;
            let mut coord2 = coord + dvec2(62.01, 62.20);

            coord1.x = coord1.x.clamp(0., 1240.);
            coord1.y = coord1.y.clamp(215., 215. + 25.4);
            coord2.x = coord2.x.clamp(0., 1240.);
            coord2.y = coord2.y.clamp(215., 215. + 25.4);

            let shape = Triangle::rectangle(coord1, coord2);

            // Record the bounding box of the shape
            let bb = shape.iter().flat_map(|x| x.array()).bounding_box();
            bbs.push(bb);

            // Record the individual triangles that make up the shape
            tris.extend(shape);
        }

        measurements.push(Measurement {
            contact:  tris,
            centroid: None,
            plane:    Plane::default(),
            offset:   Bounds::Range(-0.25, 0.25),
            last_vals: (f64::NAN, f64::NAN, f64::NAN),
            bounding_boxes: bbs,
            angle_x: Bounds::Constant(slopes_long[ii - 1] * 0.050),
            angle_y: Bounds::Range(-0.5, 0.5),
        });
    }
}*/

/*
/// Surface plate measurement strategy
fn surface_plate(measurements: &mut Vec<Measurement>) {
    let radius = 23.91 / 2.;
    //let radius = 30.;
    let inner_radius = radius - 1.9;
    let sides = 16;

    let mut raw_data = Vec::new();

    // Parse the CSV data in order
    //
    // Data is in m/m ? CSV says RAD but it matched mm/m data on the device??
    for filename in [
        "data/surface_plate_20241218/wylerUNIVERSAL data 2024-12-18 01-10-10.csv",
        "data/surface_plate_20241218/wylerUNIVERSAL data 2024-12-18 01-40-04.csv",
        "data/surface_plate_20241218/wylerUNIVERSAL data 2024-12-18 01-56-16.csv",
        "data/surface_plate_20241218/wylerUNIVERSAL data 2024-12-18 19-22-32.csv",
        "data/surface_plate_20241218/wylerUNIVERSAL data 2024-12-18 19-41-12.csv",
        "data/surface_plate_20241218/wylerUNIVERSAL data 2024-12-18 21-59-55.csv",
    ] {
        for line in std::fs::read_to_string(filename).unwrap().lines().skip(1) {
            let mut spl = line.splitn(3, ";");
            let _date = spl.next().unwrap().splitn(3, "\"").nth(1).unwrap();
            let x_slope = spl.next().unwrap().splitn(3, "\"").nth(1).unwrap();
            let y_slope = spl.next().unwrap().splitn(3, "\"").nth(1).unwrap();
            let x_slope = x_slope.parse::<f64>().unwrap() * 1000.;
            let y_slope = y_slope.parse::<f64>().unwrap() * 1000.;
            raw_data.push((x_slope, y_slope));
        }
    }
    assert!(raw_data.len() == 108 + 48*2 + 15);

    fn set1(ii: usize) -> DVec2 {
        dvec2(53.13, 53.13) +
        dvec2(((ii % 9) * 50) as f64, ((ii / 9) * 50) as f64)
    }
    fn set2(ii: usize) -> DVec2 {
        dvec2(10. * 50. + 53.13, 7. * 50. + 53.13) -
        dvec2(((ii % 9) * 50) as f64, ((ii / 9) * 50) as f64)
    }
    fn set3(ii: usize) -> DVec2 {
        dvec2(53.13, 7. * 50. + 53.13) +
        dvec2(((ii / 6) * 50) as f64, -(((ii % 6) * 50) as f64))
    }
    fn set4(ii: usize) -> DVec2 {
        dvec2(10. * 50. + 53.13, 53.13) +
        dvec2(-(((ii / 6) * 50) as f64), ((ii % 6) * 50) as f64)
    }
    fn set5(ii: usize) -> DVec2 {
        dvec2(53.13, 53.13) +
        dvec2(((ii % 5) * 100) as f64, ((ii / 5) * 100) as f64)
    }

    // (number of points, points per row, rotation)
    let sets = [
        (54, 9, 94.90, 100.00, DVec2::from_angle(0f64.to_radians()), set1 as fn(usize) -> DVec2),
        (54, 9, 94.90, 100.00, DVec2::from_angle(180f64.to_radians()), set2 as fn(usize) -> DVec2),
        (48, 6, 94.90, 150.04, DVec2::from_angle(270f64.to_radians()), set3 as fn(usize) -> DVec2),
        (48, 6, 94.90, 150.04, DVec2::from_angle(90f64.to_radians()), set4 as fn(usize) -> DVec2),
        (15, 5, 94.90, 150.04, DVec2::from_angle(0f64.to_radians()), set5 as fn(usize) -> DVec2),
    ];

    // Compute coords for data points
    let mut set_id = 0;
    let mut ii = 0;
    for &(angle_x, angle_y) in raw_data.iter() {
        let set = sets[set_id];

        if ii >= set.0 {
            set_id += 1;
            ii = 0;
        }

        let set = sets[set_id];

        // Get the origin
        let origin = set.5(ii);

        let x_coord = origin + set.4.rotate(dvec2(set.2, 0.));
        let y_coord = origin + set.4.rotate(dvec2(0., set.3));

        // Generate ranges for the X and Y slopes
        let (angle_x, angle_y) = match set_id {
            0 => (angle_x, angle_y),
            1 => (-angle_x, -angle_y),
            2 => (angle_y, -angle_x),
            3 => (-angle_y, angle_x),
            4 => (angle_x, angle_y),
            _ => unreachable!(),
        };
        let angle_x = Bounds::Constant(angle_x);
        let angle_y = Bounds::Constant(angle_y);

        let mut tris = Vec::new();
        let mut bbs = Vec::new();
        for &coord in &[origin, x_coord, y_coord] {
            let shape = Triangle::donut(coord, inner_radius, radius, sides);

            // Record the bounding box of the shape
            let bb = shape.iter().flat_map(|x| x.array()).bounding_box();
            bbs.push(bb);

            // Record the individual triangles that make up the shape
            tris.extend(shape);
        }

        measurements.push(Measurement {
            contact:  tris,
            centroid: None,
            plane:    Plane::default(),
            offset:   Bounds::Range(-0.25, 0.25),
            last_vals: (f64::NAN, f64::NAN, f64::NAN),
            bounding_boxes: bbs,
            angle_x, angle_y,
        });

        ii += 1;
    }

    for range in [0..54, 54..108, 108..156, 156..204, 204..219] {
        // Compute average X angle and average Y angle
        let mut sumx = 0.;
        let mut numx = 0.;
        let mut sumy = 0.;
        let mut numy = 0.;
        for Measurement { angle_x, angle_y, .. } in measurements[range.clone()].iter() {
            if let (Bounds::Constant(angle_x), Bounds::Constant(angle_y)) = (angle_x, angle_y) {
                sumx += *angle_x;
                sumy += *angle_y;
                numx += 1.;
                numy += 1.;
            } else {
                panic!();
            }
        }
        let avgx = sumx / numx;
        let avgy = sumy / numy;

        for Measurement { angle_x, angle_y, .. } in measurements[range].iter_mut() {
            if let (Bounds::Constant(angle_x), Bounds::Constant(angle_y)) = (angle_x, angle_y) {
                *angle_x -= avgx;
                *angle_y -= avgy;
            }
        }
    }
}*/

/// Construct window configuration
fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        sample_count: 8, // MSAA
        fullscreen: false,
        window_width: 1440,
        window_height: 900,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let it = Instant::now();

    let mut measurements: Vec<Measurement> = Vec::new();

    // Construct the measurements needed for the surface plate
    //surface_plate(&mut measurements);
    //lathe_bed_old(&mut measurements);
    lathe_bed(&mut measurements);

    println!("[{:14.6}] Generated measurements",
        it.elapsed().as_secs_f64());

    struct State {
        /// Best planes for measurements we've found so far
        best_planes: Vec<((f64, f64, f64), Plane)>,

        /// Lowest error score (error, range, xskew, yskew)
        best: (f64, f64, f64, f64),

        /// Number of iterations
        iters: u64,
    }

    // Save the current state
    let mut rng = ::rand::thread_rng();
    let state: [Arc<Mutex<State>>; SURFACES] = std::array::from_fn(|_| {
        Arc::new(Mutex::new(State {
            best_planes: measurements.iter_mut().map(|x| {
                // Randomize the measurement
                x.mutate();

                // Return the plane
                (x.last_vals, x.plane)
            }).collect::<Vec<_>>(),
            best: (f64::MAX, f64::MAX, rng.gen_range(-1.0..=1.0), rng.gen_range(-1.0..=1.0)),
            iters: 0,
        }))
    });

    println!("[{:14.6}] Created per-thread measurements with random \
              starting constraints",
        it.elapsed().as_secs_f64());

    // Take random samples of all the surfaces in the measurements
    // This is what we use to sample for minimization
    let samples = Samples::generate(&measurements, 0.1);

    // This is the set we use for rendering, it's a smaller set for faster
    // drawing
    let mut samples_render = Samples::generate(&measurements, 1.);

    println!("[{:14.6}] Generated samples",
        it.elapsed().as_secs_f64());

    // Make sure that all measurements are connected. If this is not the case,
    // the measurements will not be able to orient themselves to the same Z
    // location. Either increase samples to generate a point in a small overlap
    // or measure differently to ensure the points are connected
    let mut visited = BTreeSet::new();
    let mut connectivity =
        vec![vec![BTreeSet::new(); measurements.len()]; measurements.len()];
    let mut to_visit = vec![0];

    while let Some(meas_id) = to_visit.pop() {
        if !visited.insert(meas_id) {
            continue;
        }

        // Go through all samples contained in this measurement
        for &sample_id in samples.1[meas_id].iter() {
            // Go through all measurements used by these samples
            for &next_meas in &samples.0[sample_id].1 {
                to_visit.push(next_meas);

                // Record the connection between the two measurements
                connectivity[meas_id][next_meas].insert(sample_id);
            }
        }
    }

    let mut selected_samples = BTreeSet::new();
    for m1 in 0..measurements.len() {
        for m2 in 0..measurements.len() {
            // Get a list of all points that connect these two measurements
            let conns = &connectivity[m1][m2];

            // Require we have a decent sampling of points that connect the
            // two measurements. This just ensures we get a wider selection of
            // points.
            if conns.len() >= 20 {
                // Find the two furthest X and Y values from each other. This
                // will give us 4 points. These will be the 4 points we
                // actually use to minimize error. This will give us the four
                // most extreme points that connect the two measurements.
                //
                // The theory is that since the two measurements are planes,
                // the lines connecting the furthest X and furthest Y values
                // must provide the most extreme errors as the Z values will
                // diverge the most at the extremes.
                let mut min_x = (f64::MAX, 0);
                let mut max_x = (f64::MIN, 0);
                let mut min_y = (f64::MAX, 0);
                let mut max_y = (f64::MIN, 0);
                for &sample_id in conns {
                    // Get the position of the sample
                    let pos = samples.0[sample_id].0;

                    if pos.x < min_x.0 { min_x = (pos.x, sample_id); }
                    if pos.x > max_x.0 { max_x = (pos.x, sample_id); }
                    if pos.y < min_y.0 { min_y = (pos.y, sample_id); }
                    if pos.y > max_y.0 { max_y = (pos.y, sample_id); }
                }

                // Record the samples
                selected_samples.insert(min_x.1);
                selected_samples.insert(max_x.1);
                selected_samples.insert(min_y.1);
                selected_samples.insert(max_y.1);
            } else {
                assert!(conns.len() == 0, "Connectivity is not strong enough \
                    between measurements {m1} and {m2}");
            }
        }
    }
    let selected_samples = selected_samples.into_iter().collect::<Vec<_>>();
    println!("[{:14.6}] Reduced to {} samples",
        it.elapsed().as_secs_f64(), selected_samples.len());

    if visited.len() != measurements.len() {
        panic!(
        "WARNING: Measurements were not fully connected. Increase \
         measurements until \
         all measurements can be reached through traversing through points.
         Measurements {} | Visited {}", measurements.len(), visited.len());
    }

    for state in state.iter() {
        let mut measurements = measurements.clone();
        let state = state.clone();
        let samples = samples.clone();
        let selected_samples = selected_samples.clone();

        std::thread::spawn(move || {
            let mut rng = ::rand::thread_rng();

            loop {
                let (mut xskew, mut yskew) = {
                    let mut state = state.lock().unwrap();

                    // Restore best parameters
                    measurements.iter_mut().enumerate()
                            .for_each(|(ii, x)| {
                        x.last_vals = state.best_planes[ii].0;
                        x.plane = state.best_planes[ii].1
                    });
                    state.iters += 1;
                    (state.best.2, state.best.3)
                };

                let mut skew = true;
                if ::rand::random::<usize>() % 4 == 0 {
                    // Mutate skews
                    if ::rand::random::<bool>() {
                        xskew += rng.gen_range(-0.0001..=0.0001);
                    } else {
                        yskew += rng.gen_range(-0.0001..=0.0001);
                    }
                    skew = true;
                } else {
                    // Randomly mutate some measurements
                    for _ in 0..::rand::random::<usize>() % 2 + 1 {
                        let sel = ::rand::random::<usize>() % measurements.len();
                        measurements[sel].mutate();
                    }
                }

                // Recompute all points (skews need this)
                for meas in measurements.iter_mut() {
                    meas.recompute(xskew, yskew);
                }

                // Compute the overlap distances
                let mut sum_dist = 0.;
                let mut num_dist = 0.;
                let mut min_point = f64::MAX;
                let mut max_point = f64::MIN;
                for &sample_id in &selected_samples {
                    let &(point, ref deps, _) = &samples.0[sample_id];
                    if !deps.is_empty() {
                        let mut min = f64::MAX;
                        let mut max = f64::MIN;
                        for &dep in deps {
                            let m = measurements[dep].plane.get(point);
                            min = min.min(m.z);
                            max = max.max(m.z);
                        }

                        min_point = min_point.min(min);
                        max_point = max_point.max(max);

                        sum_dist += (max - min).abs();
                        num_dist += 1.;
                    }
                }
                let avg_dist = sum_dist / num_dist;
                let range = max_point - min_point;

                // Recompute all points to display the _original_ level
                // readings!
                //
                // This is for viewing if the lathe bed is out of level and
                // needs to be re-leveled
                //for meas in measurements.iter_mut() { meas.recompute(0., 0.); }

                {
                    let mut state = state.lock().unwrap();

                    if skew && range < state.best.1 {
                        // Save the best planes
                        state.best_planes.iter_mut().enumerate()
                                .for_each(|(ii, x)| {
                            *x = (measurements[ii].last_vals, measurements[ii].plane);
                        });

                        state.best = (avg_dist, range, xskew, yskew);
                    }

                    if (avg_dist, range, xskew, yskew) < state.best {
                        // Save the best planes
                        state.best_planes.iter_mut().enumerate()
                                .for_each(|(ii, x)| {
                            *x = (measurements[ii].last_vals, measurements[ii].plane);
                        });

                        state.best = (avg_dist, range, xskew, yskew);
                    }
                }
            }
        });
    }

    println!("[{:14.6}] Starting rendering process",
        it.elapsed().as_secs_f64());

    // Create renderer
    let mut renderer = Renderer::default();

    let mut should_draw_text = true;
    let mut should_draw_feet = true;
    let mut solutions = Vec::new();
    let it = Instant::now();
    for frame in 1u64.. {
        use std::fmt::Write;

        // Toggle drawing text
        if is_key_pressed(KeyCode::T) {
            should_draw_text = !should_draw_text;
        }
        if is_key_pressed(KeyCode::F) {
            should_draw_feet = !should_draw_feet;
        }

        // Render the best measurements we've had so far
        clear_background(DARKGRAY);

        set_default_camera();

        let mut msg = String::new();

        solutions.clear();
        for state in state.iter() {
            let (score, iters) = {
                let state = state.lock().unwrap();

                // Restore best parameters
                measurements.iter_mut().enumerate()
                        .for_each(|(ii, x)| {
                    x.last_vals = state.best_planes[ii].0;
                    x.plane = state.best_planes[ii].1;
                });

                (state.best, state.iters)
            };

            // Update samples
            samples_render.recompute(&measurements);
            samples_render.simplify();

            // Compute statistics for all points in this set
            // (min, max, centroid)
            let mut range = (f64::MAX, f64::MIN);
            let mut sum = dvec3(0., 0., 0.);
            let mut n   = 0.;
            for pt in samples_render.0.iter().flat_map(|(pt, _, (s, n))| if *n > 0. { Some(pt.extend(s / n)) } else { None }) {
                range.0 = range.0.min(pt.z);
                range.1 = range.1.max(pt.z);
                sum += pt;
                n   += 1.;
            }

            // Save solution
            solutions.push((score, iters, samples_render.clone(), range, sum, n));
        }

        // Get global minimum and maximum Z values as well as the centroid of
        // all points
        let mut range = (f64::MAX, f64::MIN);
        let mut sum = dvec3(0., 0., 0.);
        let mut n   = 0.;
        for (_, _, _, srange, ssum, sn) in solutions.iter_mut() {
            range.0 = range.0.min(srange.0);
            range.1 = range.1.max(srange.1);

            sum += *ssum;
            n   += *sn;
        }
        let centroid = sum / n;

        // Camera target is the centroid
        let target = (centroid * dvec3(1., 1., Z_SCALE)).as_vec3();

        let camera_angle = (frame as f32 * 1.) % 360.;
        let camera_angle: f32 = 0.;
        let camera_dist = 800.;

        // Start at [0, -dist], which is "standing in front of the surface
        // plate"
        let camera_x_off = camera_dist * camera_angle.to_radians().sin();
        let camera_y_off = -camera_dist * camera_angle.to_radians().cos();

        let cam = Camera3D {
            position: vec3(target.x + camera_x_off, target.y + camera_y_off, target.z + 600.),
            up: vec3(0., 0., 1.),
            target,
            ..Default::default()
        };
        set_camera(&cam);

        // Draw all solutions
        for (score, iters, samples_render, _, _, _) in solutions.iter_mut() {
            samples_render.draw(&mut renderer, &visited, range.0, range.1);

            writeln!(&mut msg, "Score ({:10.6}, {:10.6}, {:10.6}, {:10.6}) um err/point | Iters {:10.0}/sec | Samples {}",
                score.0 * 1e3, score.1 * 1e3,
                score.3, -score.2, // Intentionally flip the X and Y axis, since that's what we did
                                  // with the raw input measurements
                *iters as f64 / it.elapsed().as_secs_f64(),
                samples_render.0.len()).unwrap();
        }

        // Draw all measurement bounding boxes
        let mut bb_labels = Vec::new();
        if should_draw_feet {
            for (ii, meas) in measurements.iter().enumerate() {
                for bb in &meas.raw_bounding_boxes {
                    // Get the bottom left and top right coordinates
                    let bl = bb.0;
                    let tr = bb.1;

                    //    +----+ tr
                    //    |    |
                    //    |    |
                    // bl +----+

                    let color = if ii & 1 == 0 { BLACK } else { WHITE };

                    let mut avg = dvec3(0., 0., 0.);
                    for (a, b) in [
                        (dvec2(bl.x, bl.y), dvec2(tr.x, bl.y)),
                        (dvec2(tr.x, bl.y), dvec2(tr.x, tr.y)),
                        (dvec2(tr.x, tr.y), dvec2(bl.x, tr.y)),
                        (dvec2(bl.x, tr.y), dvec2(bl.x, bl.y)),
                    ] {
                        // Get the Z positions of the level contact feet
                        let a = (meas.plane.get(a) +
                            dvec3(0., 0., samples_render.2)) *
                            dvec3(1., 1., Z_SCALE);
                        let b = (meas.plane.get(b) +
                            dvec3(0., 0., samples_render.2)) *
                            dvec3(1., 1., Z_SCALE);

                        avg += a;
                        avg += b;

                        draw_line_3d(a.as_vec3(), b.as_vec3(), color);
                    }

                    // Compute centroid for the bounding box
                    let centroid = avg / 8.;

                    let target = cam.matrix().project_point3(
                        centroid.as_vec3());

                    let screen_target = vec2(
                        (target.x + 1.) / 2. * screen_width(),
                        screen_height() - (target.y + 1.) / 2. * screen_height(),
                    );

                    bb_labels.push((format!("M{:03}", ii + 1), screen_target, color));
                }
            }
        }

        set_default_camera();

        let mut text_to_draw = Vec::new();
        if should_draw_text {
            for (_, _, samples_render, _, _, _) in solutions.iter() {
                for pos in samples_render.0.iter().flat_map(|(pt, _, (s, n))| if *n > 0. { Some(pt.extend(s / n)) } else { None }) {
                    let target = cam.matrix().project_point3(
                        (pos * dvec3(1., 1., Z_SCALE)).as_vec3());

                    let screen_target = vec2(
                        (target.x + 1.) / 2. * screen_width(),
                        screen_height() - (target.y + 1.) / 2. * screen_height(),
                    );

                    text_to_draw.push(
                        (pos, screen_target.x, screen_target.y));
                }
            }
        }

        if should_draw_feet {
            for (text, pos, col) in bb_labels {
                draw_text(&text, pos.x, pos.y, 16., col);
            }
        }

        for (ii, meas) in measurements.iter().enumerate() {
            // 1-indexed measurement ID
            let mid = ii + 1;
            let color = if ii & 1 == 0 { BLACK } else { WHITE };

            let x = ((ii / 16) as f32) * 400. + 2.;
            let y = ((ii % 16) as f32) * 16.  + screen_height() - 256.;

            // Flip angles since we use a weird coord system
            let Bounds::Constant(angle_x) = meas.angle_y else { panic!() };
            let Bounds::Constant(angle_y) = meas.angle_x else { panic!() };
            let angle_x = -angle_x;

            let (score, _, _, _, _, _) = solutions.last().unwrap();
            let simx = angle_x - score.3;
            let simy = angle_y - -score.2;

            draw_text(&format!("M{mid:03} {angle_x:6.3} {angle_y:6.3} -> {simx:6.3} {simy:6.3}"), x, y, 16., color);
        }

        // Reduce the amount of text on screen by picking the average for a
        // given area
        let area = 100.;
        let mut sums = std::collections::BTreeMap::new();
        for &(pos, x, y) in &text_to_draw {
            let sx = (pos.x / area).round() as i64;
            let sy = (pos.y / area).round() as i64;
            sums.entry((sx, sy))
                .or_insert_with(Vec::new)
                .push(dvec3(x as f64, y as f64, pos.z));
        }

        for (_, values) in sums.iter_mut() {
            values.sort_by(|a, b| b.z.partial_cmp(&a.z).unwrap());

            for (idx, offset) in [
                (0, -25.),
                (values.len() - 1, 25.),
            ] {
                let pt = values[idx];
                let text = format!("{:.3} um", pt.z * 1e3);
                let dims = measure_text(&text, None, 16, 1.0);

                draw_text(&text,
                    (pt.x as f32 - dims.width / 2.).round(),
                    (pt.y as f32 + dims.height / 2. + offset).round(),
                    16.,
                    Renderer::color((pt.z - range.0) / (range.1 - range.0)));
            }
        }

        writeln!(&mut msg,
            "FPS: {}", frame as f64 / it.elapsed().as_secs_f64()).unwrap();

        draw_multiline_text(&msg, 0., 80., 16., Some(1.), BLACK);

        let width = screen_width() as usize;

        draw_rectangle(0., 5., width as f32, 30., BLACK);

        for pixel in 0..width {
            let normal = pixel as f64 / width as f64;
            draw_rectangle(pixel as f32, 10., 1., 20., Renderer::color(normal));

            if pixel % 128 == 0 {
                draw_text(&format!("{:.2} um", (normal * range.1 + range.0) * 1e3), pixel as f32, 55., 16., Renderer::color(normal));
            }
        }

        let image = get_screen_data();
        image.export_png("screenshot.png");

        next_frame().await;
    }
}

