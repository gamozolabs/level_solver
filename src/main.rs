// cargo add macroquad nalgebra rand

#![feature(array_windows)]

mod color;

use std::f64::consts::*;
use std::sync::{Arc, Mutex};
use ::rand::prelude::*;
use macroquad::prelude::*;

/// Number of surfaces to create (also number of threads)
const SURFACES: usize = 8;

/// Extra Z-scale multiplication for rendering only. This is to visually
/// amplify the Z axis.
const Z_SCALE: f64 = 40000.;

#[derive(Clone)]
struct Samples(
    /// Random sample points and the measurements that contain the points
    /// (xy, dependencies, (sum, count))
    Vec<(DVec2, Vec<usize>, (f64, f64))>
);

impl Samples {
    /// Generate random set of samples for `measurements`
    fn generate(measurements: &[Measurement], num_samples: usize) -> Self {
        let mut samples = Vec::new();

        // Randomly sample all triangles
        for meas in measurements {
            for tri in &meas.contact {
                for pos in tri.sample(num_samples) {
                    samples.push((pos, Vec::new(), (0., 0.)));
                }
            }
        }

        // Determine which measurements contain the sample points
        for (point, containing, _) in samples.iter_mut() {
            for (ii, meas) in measurements.iter().enumerate() {
                for tri in &meas.contact {
                    if tri.contains(*point) {
                        containing.push(ii);
                    }
                }
            }
        }

        // Randomize sample ordering
        samples.shuffle(&mut ::rand::thread_rng());

        Self(samples)
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

    /// Finds the average plane through all the points and normalizes all
    /// the data to this plane
    fn simplify(&mut self) {
        // Generate matrix from data
        let mut data = nalgebra::DMatrix::zeros(self.0.len(), 3);
        for (ii, (pt, _, (s, n))) in self.0.iter().enumerate() {
            data[(ii, 0)] = pt.x;
            data[(ii, 1)] = pt.y;
            data[(ii, 2)] = *s / *n;
        }

        // Compute centroid
        let centroid = data.row_mean();

        // Subtract centroid
        data.row_iter_mut().for_each(|mut x| x -= &centroid);

        // Compute SVD
        let svd = data.svd(false, true);
        let left = svd.v_t.unwrap();

        // Get the right least singular value
        let normal = left.row_iter().last().unwrap();

        // Create the best fitting plane through the data
        let plane = Plane {
            normal: dvec3(normal[0], normal[1], normal[2]),
            offset: 0.,
        };

        // Adjust all data to this plane
        let mut min = f64::MAX;
        for (loc, _, (sum, n)) in self.0.iter_mut() {
            *sum -= plane.get(*loc).z * *n;

            let val = *sum / *n;
            min = min.min(val);
        }

        // Adjust all data to zero
        for (_, _, (sum, n)) in self.0.iter_mut() {
            *sum -= min * *n;
        }
    }

    /// Draw the sample points
    fn draw(&mut self, renderer: &mut Renderer, measurements: &[Measurement])
            -> (f64, f64) {
        // Find the extents of the data
        let mut min_z = f64::MAX;
        let mut max_z = f64::MIN;
        for &(point, _, (s, n)) in self.0.iter() {
            let z = s / n;
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        let range_z = max_z - min_z;

        // Display the data
        for &(loc, _, (s, n)) in self.0.iter() {
            let z = s / n;
            let pct = (z - min_z) / range_z;
            let col = Renderer::color(pct);
            renderer.draw_square(loc.extend(z), 2., col);
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
#[derive(Clone, Copy)]
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

    /// Compute the bounding box for the surface
    #[allow(dead_code)]
    fn bounding_box(&self) -> (DVec2, DVec2) {
        let mut xmin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;

        for tri in &self.contact {
            for vertex in tri.array() {
                xmin = xmin.min(vertex.x);
                ymin = ymin.min(vertex.y);
                xmax = xmax.max(vertex.x);
                ymax = ymax.max(vertex.y);
            }
        }

        (dvec2(xmin, ymin), dvec2(xmax, ymax))
    }

    /// Randomly sample the internal bounds and regenerate the plane
    fn mutate(&mut self) {
        // Pick random values for the measurements
        let angle_x = self.angle_x.sample();
        let angle_y = self.angle_y.sample();
        let offset  = self.offset.sample();

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
}

/// Surface plate measurements 2024-12-10, quick 30 minute sampling at 50mm
/// steps before NYC flight
fn surface_plate(measurements: &mut Vec<Measurement>) {
    // Surface plate layout
    //
    // |     613 mm     |
    // +----------------+ -
    // |                |
    // |                | 459 mm
    // |                |
    // +----------------+ -
    //
    // There are 3mm margins on the plate (the above dimensions have 3mm of
    // taper on all edges, thus the actual flat part of the plate removes 3mm
    // on each side)
    //
    // The data is sampled with the 2d level at 50mm increments starting from
    // the bottom left, going left to right, bottom to top
    //
    // The 2d level base is set at 150mm on X and Y. With 23.91mm diameter
    // round feet.
    //
    // The 50mm x 50mm grid is made with a 600mm ruler centered on the plate.
    // this adds another 3.5mm of margins on the flat part of the plate
    //
    // Readings are in mm/m. Positive readings are a raised "right/top" side.
    let data = &[
        (0.010, -0.140), (0.008, -0.145), (0.008, -0.150), (0.007, -0.154),
        (0.003, -0.157), (0.001, -0.160), (-0.001, -0.162), (-0.003, -0.162),

        (0.005, -0.141), (0.005, -0.145), (0.004, -0.148), (0.003, -0.151),
        (0.001, -0.153), (0.001, -0.155), (0.000, -0.158), (0.000, -0.158),

        /*(-0.010, -0.157), Extra data point off grid */ 

        (0.000, -0.143), (0.001, -0.143), (0.001, -0.144),
        (0.000, -0.146), (0.000, -0.148), (0.000, -0.149), (0.000, -0.151),
        (0.000, -0.153),

        /* (-0.005, -0.152), Extra data point off grid */

        (-0.002, -0.145), (-0.001, -0.142),
        (-0.001, -0.142), (-0.002, -0.142), (-0.002, -0.142), (-0.002, -0.144),
        (-0.001, -0.146), (0.000, -0.147),

        (-0.004, -0.150), (-0.003, -0.143),
        (-0.003, -0.140), (-0.004, -0.139), (-0.004, -0.140), (-0.004, -0.141),
        (-0.003, -0.143), (-0.001, -0.146),
    ];

    // Extra data for the top right corner of the surface plate that was not
    // touched by any measurement. The level was turned 90 degrees for these
    // samples
    let special_data = &[
        (0.142, -0.003), (0.144, -0.002), (0.148, 0.001),
        (0.142, -0.006), (0.144, -0.002), (0.149, 0.001),
        (0.144, -0.007), (0.145, -0.001), (0.149, 0.003),
    ];

    // All y points are negative from our data, just assert valid data entry
    for &(x, y) in data {
        assert!(y < 0.);
    }

    //let radius = 23.91 / 2.;
    let radius = 80. / 2.;
    let sides = 36;
    let circle = true;

    // Compute coords for data points
    'next: for (ii, &(angle_x, angle_y)) in data.iter().enumerate() {
        // y_coord
        // ^
        // ^
        // ^
        // o > > > x_coord
        // origin
        let origin = dvec2(((ii % 8 + 1) * 50) as f64,
            ((ii / 8 + 1) * 50) as f64);

        println!("{origin}");

        let x_coord = origin + dvec2(150., 0.);
        let y_coord = origin + dvec2(0., 150.);

        // Default measurement uncertainty to apply to angles (mm/m)
        let uncertainty = 0.0;

        // Generate ranges for the X and Y slopes
        let angle_x = Bounds::Range(angle_x - uncertainty, angle_x + uncertainty);
        let angle_y = Bounds::Range(angle_y - uncertainty, angle_y + uncertainty);

        for &coord in &[origin, x_coord, y_coord] {
            if coord.x % 150. != 50. || coord.y % 150. != 50. {
                //println!("FILTER");
                //continue 'next;
            }
        }

        let mut tris = Vec::new();
        for &coord in &[origin, x_coord, y_coord] {
            if circle {
                tris.extend(Triangle::polygon(coord, radius, sides));
            } else {
                let bl = coord - dvec2(radius, radius);
                let tr = coord + dvec2(radius, radius);

                tris.extend(Triangle::rectangle(bl, tr));
            }
        }

        measurements.push(Measurement {
            contact:  tris,
            centroid: None,
            plane:    Plane::default(),
            offset:   Bounds::Range(-0.25, 0.25),
            angle_x, angle_y,
        });
    }

    println!();

    // Compute coords for special data points
    //
    // These are rotated clockwise 90 degrees
    'next: for (ii, &(angle_x, angle_y)) in special_data.iter().enumerate() {
        // y_coord
        // ^
        // ^
        // ^
        // o > > > x_coord
        // origin
        let origin = dvec2(((ii / 3 + 6) * 50) as f64,
            ((ii % 3) as isize * -50) as f64 + 400.);

        println!("{origin}");

        let x_coord = origin + dvec2(150., 0.);
        let y_coord = origin - dvec2(0., 150.);

        // Default measurement uncertainty to apply to angles (mm/m)
        let uncertainty = 0.0;

        let (angle_x, angle_y) = (angle_y, -angle_x);

        // Generate ranges for the X and Y slopes
        let angle_x = Bounds::Range(angle_x - uncertainty, angle_x + uncertainty);
        let angle_y = Bounds::Range(angle_y - uncertainty, angle_y + uncertainty);

        for &coord in &[origin, x_coord, y_coord] {
            if coord.x % 150. != 50. || coord.y % 150. != 50. {
                //println!("FILTER");
                //continue 'next;
            }
        }

        let mut tris = Vec::new();
        for &coord in &[origin, x_coord, y_coord] {
            if circle {
                tris.extend(Triangle::polygon(coord, radius, sides));
            } else {
                let bl = coord - dvec2(radius, radius);
                let tr = coord + dvec2(radius, radius);

                tris.extend(Triangle::rectangle(bl, tr));
            }
        }

        measurements.push(Measurement {
            contact:  tris,
            centroid: None,
            plane:    Plane::default(),
            offset:   Bounds::Range(-0.25, 0.25),
            angle_x, angle_y,
        });
    }
}

/// Construct window configuration
fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        sample_count: 8, // MSAA
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut measurements: Vec<Measurement> = Vec::new();

    // Construct the measurements needed for the surface plate
    surface_plate(&mut measurements);

    struct State {
        /// Best planes for measurements we've found so far
        best_planes: Vec<Plane>,

        /// Lowest error score
        best: f64,

        /// Number of iterations
        iters: u64,
    }

    // Compute overlapping rectangles
    let mut overlaps = Vec::new();
    for m1 in 0..measurements.len() {
        for m2 in 0..measurements.len() {
            let m1b = measurements[m1].bounding_box();
            let m2b = measurements[m2].bounding_box();

            // Compute overlapping rectangle coords
            let bl = dvec2(
                m1b.0.x.max(m2b.0.x),
                m1b.0.y.max(m2b.0.y),
            );
            let tr = dvec2(
                m1b.1.x.min(m2b.1.x),
                m1b.1.y.min(m2b.1.y),
            );

            if tr.x >= bl.x && tr.y >= bl.y {
                // If there is overlap, record the overlapping box and the two
                // measurements that overlap
                overlaps.push(([
                    bl,
                    tr,
                    dvec2(bl.x, tr.y),
                    dvec2(tr.x, bl.y),
                ], m1, m2));
            }
        }
    }

    // Save the current state
    let state: [Arc<Mutex<State>>; SURFACES] = std::array::from_fn(|_| {
        Arc::new(Mutex::new(State {
            best_planes: measurements.iter_mut().map(|x| {
                // Randomize the measurement
                x.mutate();

                // Return the plane
                x.plane
            }).collect::<Vec<Plane>>(),
            best: f64::MAX,
            iters: 0,
        }))
    });

    // Take random samples of all the surfaces in the measurements
    let mut samples = Samples::generate(&measurements, 1);

    for state in state.iter() {
        let mut measurements = measurements.clone();
        let state = state.clone();
        let samples = samples.clone();
        let overlaps = overlaps.clone();

        std::thread::spawn(move || {
            loop {
                {
                    let mut state = state.lock().unwrap();

                    // Restore best parameters
                    measurements.iter_mut().enumerate()
                        .for_each(|(ii, x)| x.plane = state.best_planes[ii]);
                    state.iters += 1;
                }

                // Randomly mutate some measurements
                for _ in 0..::rand::random::<usize>() % 2 + 1 {
                    let sel = ::rand::random::<usize>() % measurements.len();
                    measurements[sel].mutate();
                }

                /*
                // Compute the overlap distances
                let mut sum_dist = 0.;
                let mut num_dist = 0.;
                for &(points, m1, m2) in &overlaps {
                    for point in points {
                        let z1 = measurements[m1].plane.get(point).z;
                        let z2 = measurements[m2].plane.get(point).z;
                        sum_dist += (z2 - z1).abs();
                        num_dist += 1.;
                    }
                }
                let avg_dist = sum_dist / num_dist;*/

                // Compute the overlap distances
                let mut sum_dist = 0.;
                let mut num_dist = 0.;
                for &(point, ref deps, _) in &samples.0 {
                    let mut min = f64::MAX;
                    let mut max = f64::MIN;
                    for &dep in deps {
                        min = min.min(measurements[dep].plane.get(point).z);
                        max = max.max(measurements[dep].plane.get(point).z);
                    }

                    sum_dist += (max - min).abs();
                    num_dist += 1.;
                }
                let avg_dist = sum_dist / num_dist;

                {
                    let mut state = state.lock().unwrap();

                    if avg_dist < state.best {
                        // Save the best planes
                        state.best_planes.iter_mut().enumerate()
                            .for_each(|(ii, x)| *x = measurements[ii].plane);

                        state.best = avg_dist;
                    }
                }
            }
        });
    }

    // Create renderer
    let mut renderer = Renderer::default();

    let it = std::time::Instant::now();
    for frame in 1u64.. {
        use std::fmt::Write;

        // Render the best measurements we've had so far
        clear_background(DARKGRAY);

        let mut msg = String::new();
        let mut range = (0., 0.);
        for state in state.iter() {
            let (score, iters) = {
                let state = state.lock().unwrap();

                // Restore best parameters
                measurements.iter_mut().enumerate()
                    .for_each(|(ii, x)| x.plane = state.best_planes[ii]);

                (state.best, state.iters)
            };

            // Update samples
            samples.recompute(&measurements);
            samples.simplify();

            // Average the data
            let mut sum = dvec3(0., 0., 0.);
            let mut cnt = 0.;
            for pt in samples.0.iter().map(|(pt, _, (s, n))| pt.extend(s / n)) {
                sum += pt;
                cnt += 1.;
            }
            let avg = sum / cnt;

            // Camera target is the centroid
            let target = vec3(avg.x as f32, avg.y as f32, (avg.z * Z_SCALE) as f32);

            set_camera(&Camera3D {
                position: vec3(target.x, target.y - 600., target.z + 600.),
                up: vec3(0., 0., 1.),
                target,
                ..Default::default()
            });

            let (min, max) = samples.draw(&mut renderer, &measurements);

            writeln!(&mut msg, "Score {:7.3} um err/point | Range {:7.3} um | Iters {:10.0}/sec",
                score * 1e3,
                (max - min) * 1e3,
                iters as f64 / it.elapsed().as_secs_f64()).unwrap();
            range = (min, max - min);
        }

        writeln!(&mut msg,
            "FPS: {}", frame as f64 / it.elapsed().as_secs_f64()).unwrap();

        set_default_camera();
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

        next_frame().await;
    }
}

