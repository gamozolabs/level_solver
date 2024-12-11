#![feature(float_next_up_down)]

mod color;

use std::f64::consts::*;
use std::sync::{Arc, Mutex};
use ::rand::prelude::*;
use macroquad::prelude::*;

/// Extra Z-scale multiplication for rendering only. This is to visually
/// amplify the Z axis.
const Z_SCALE: f64 = 4000.;

/// A renderer which can cache mesh allocations
struct Renderer {
    /// Internal mesh used during rendering. Used just to prevent reallocs of
    /// backing buffers
    mesh: Mesh,

    /// Random sample points and the measurements that contain the points
    samples: Vec<(DVec2, Vec<usize>, f64)>,

    min_z: f64,
    max_z: f64,
}

impl Default for Renderer {
    fn default() -> Self {
        Self {
            mesh: Mesh {
                vertices: Vec::new(),
                indices:  Vec::new(),
                texture:  None,
            },
            samples: Vec::new(),
            min_z: 0.,
            max_z: 0.,
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

        // Compute the normals to update the colors
        for triangle in self.mesh.vertices.chunks_mut(3) {
            // Scale Z to amplify the surface
            for vertex in &mut *triangle {
                vertex.position.z *= Z_SCALE as f32;
            }

            // Compute the normal
            let normal = (triangle[1].position - triangle[0].position)
                .cross(triangle[2].position - triangle[0].position)
                .normalize();

            // Rotate the normal a bit for better lighting contrast
            let normal = Mat3::from_rotation_y(0.2) *
                Mat3::from_rotation_x(0.2) * normal;

            // Compute the shading based on the normal
            let color = Vec4::new(normal.z, normal.z, normal.z, 1.);

            // Update the colors with the shading of the normal
            for vertex in triangle {
                vertex.color = Color::from_vec(Color::from_rgba(
                    vertex.color[0], vertex.color[1], vertex.color[2],
                    vertex.color[3]).to_vec() * color).into();
            }
        }

        // Render the mesh!
        draw_mesh(&self.mesh);
    }

    /// Get the color for a given normalized value [0.0, 1.0]
    fn color(val: f64) -> Color {
        // Inferno color palette lookup
        let idx = (val * 254.99) as usize;
        let partial = (val * 254.99) - idx as f64;
        let col1 = DVec3::from_array(color::INFERNO[idx + 0]);
        let col2 = DVec3::from_array(color::INFERNO[idx + 1]);
        let col = col1 * (1. - partial) + col2 * partial;
        Color::new(col[0] as f32, col[1] as f32, col[2] as f32, 1.)
    }

    fn draw_square(&mut self, center: DVec3, size: f64, color: Color) {
        // Create two triangles and draw them as a square
        let bl = center + dvec3(-size / 2., -size / 2., 0.);
        let br = center + dvec3( size / 2., -size / 2., 0.);
        let tl = center + dvec3(-size / 2.,  size / 2., 0.);
        let tr = center + dvec3( size / 2.,  size / 2., 0.);

        // Generate both CCW triangles
        for vert in [bl, tr, tl, bl, br, tr] {
            self.mesh.vertices.push(Vertex::new(
                vert.x as f32, vert.y as f32, vert.z as f32, 0., 0., color));
        }

        if self.mesh.vertices.len() >= 4096 {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.mesh.vertices.len() > 0 {
            self.draw_int();
            self.mesh.vertices.clear();
        }
    }

    /// Generate random samples for `measurements`
    fn generate_samples(&mut self, measurements: &[Measurement],
            samples: usize) {
        // Generate the random samples
        if self.samples.len() == 0 {
            // Randomly sample all triangles
            for meas in measurements {
                for tri in &meas.contact {
                    for pos in tri.sample(samples) {
                        self.samples.push((pos, Vec::new(), f64::MAX));
                    }
                }
            }

            // Determine which measurements contain the sample points
            for (point, containing, _) in self.samples.iter_mut() {
                for (ii, meas) in measurements.iter().enumerate() {
                    for tri in &meas.contact {
                        if tri.contains(*point) {
                            containing.push(ii);
                        }
                    }
                }
            }

            // Randomize sample ordering
            self.samples.shuffle(&mut ::rand::thread_rng());
        }

        // Reset sample points to max
        self.samples.iter_mut().for_each(|(_, _, v)| *v = f64::MAX);

        // Sample all points, looking for the minimum value for a plane that
        // contains the point
        for (point, containing, val) in self.samples.iter_mut() {
            for meas_id in containing {
                *val = val.min(measurements[*meas_id].plane.get(*point).z);
            }
        }
    }

    /// Finds the average plane through all the points and normalizes all
    /// the data to this plane
    fn simplify(&mut self) {
        // Generate matrix from data
        let mut data = nalgebra::DMatrix::zeros(self.samples.len(), 3);
        for (ii, (pt, _, z)) in self.samples.iter().enumerate() {
            data[(ii, 0)] = pt.x;
            data[(ii, 1)] = pt.y;
            data[(ii, 2)] = *z;
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
        let mut min = std::f64::MAX;
        for (loc, _, z) in self.samples.iter_mut() {
            *z -= plane.get(*loc).z;
            min = min.min(*z);
        }

        // Adjust all data to zero
        for (_, _, z) in self.samples.iter_mut() {
            *z -= min;
        }
    }

    fn draw(&mut self) {
        // Find the extents of the data
        self.min_z = f64::MAX;
        self.max_z = f64::MIN;
        for (_, _, z) in self.samples.iter() {
            self.min_z = self.min_z.min(*z);
            self.max_z = self.max_z.max(*z);
        }

        let range_z = self.max_z - self.min_z;

        // Display the data
        let mut samples = Vec::new();
        std::mem::swap(&mut self.samples, &mut samples);
        for (loc, _, sample) in samples.iter() {
            let pct = (*sample - self.min_z) / range_z;
            let col = Self::color(pct);
            self.draw_square(loc.extend(*sample), 1., col);
        }
        std::mem::swap(&mut self.samples, &mut samples);

        // Flush any pending triangles
        self.flush();
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
            samples.push((1. - r1.sqrt()) * self.a +
                (r1.sqrt() * (1. - r2)) * self.b +
                (r2 * r1.sqrt()) * self.c);
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

    /// Center of mass for the contact patch. This is considered the origin
    /// for the `plane`
    center_of_mass: Option<DVec2>,

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
    /// Compute and/or fetch the center of mass for the contact points of the
    /// measurement
    fn center_of_mass(&mut self) -> DVec2 {
        // Get the center of mass
        if let Some(com) = self.center_of_mass {
            com
        } else {
            // Compute the center of mass
            let mut com = dvec2(0., 0.);
            for tri in &self.contact {
                com += tri.a;
                com += tri.b;
                com += tri.c;
            }
            com /= (self.contact.len() * 3) as f64;
            self.center_of_mass = Some(com);
            com
        }
    }

    /// Compute the bounding box for the surface
    fn bounding_box(&self) -> (DVec2, DVec2) {
        let mut xmin = std::f64::MAX;
        let mut xmax = std::f64::MIN;
        let mut ymin = std::f64::MAX;
        let mut ymax = std::f64::MIN;

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
        let angle_x = self.angle_x.sample();
        let angle_y = self.angle_y.sample();
        let offset  = self.offset.sample();

        let com = self.center_of_mass();

        let x_solve = DVec2::from_angle(FRAC_PI_2)
            .rotate(dvec2(1000., angle_x * 2.));
        let y_solve = DVec2::from_angle(FRAC_PI_2)
            .rotate(dvec2(1000., angle_y * 2.));

        // Compute the normal for the plane and normalize the resulting vector
        self.plane.normal = (dvec3(x_solve.x, 0., x_solve.y) +
            dvec3(0., y_solve.x, y_solve.y)).normalize();

        // Clear the offset so we can solve for the new one
        self.plane.offset = 0.;

        // Apply the offset at the center of mass rather than the origin
        self.plane.offset = offset - self.plane.get(com).z;

        #[cfg(debug_assertions)]
        {
            // Validate that our computed plane produces the desired angles
            let x_reading = self.plane.get(dvec2(1000., 0.)).z -
                self.plane.get(dvec2(0., 0.)).z;
            let y_reading = self.plane.get(dvec2(0., 1000.)).z -
                self.plane.get(dvec2(0., 0.)).z;
            assert!((angle_x - x_reading).abs() < 0.0000001);
            assert!((angle_y - y_reading).abs() < 0.0000001);
        }
    }
}

/// Construct window configuration
fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        fullscreen:   false,
        sample_count: 8, // MSAA
        platform: miniquad::conf::Platform {
            // Set to None for vsync on, Some(0) for off
            swap_interval: None,
            //swap_interval: Some(0),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
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

    // Our desired hats
    let mut raw_hats = Vec::new();

    let across_data = [-0.5_f64,-0.5,0.,-0.5,-1.,0.,-1.,-0.5,0.,0.5,0.,0.,1.,1.,1.,1.,1.5,1.5,1.5,1.5,2.,2.5,2.,2.5,];
    let slopes_long = [1f64,0.,-0.5,-0.5,-1.,-1.,-1.,-1.,-0.5,-0.5,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-2.,-2.,-2.,-2.,-2.,-1.5,-2.,-2.,-2.,];
    let slopes_short = [0f64,0.,0.,-0.5,0.,0.,0.,0.,0.,-0.5,0.,-0.5,0.,-0.5,-0.5,0.,-0.5,0.,-0.5,-1.,];

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

        let x1 = x1.max(0.);
        let x2 = x2.min(1005.);

        let target = -across_data[ii - 1];

        raw_hats.push(((f64::NAN, target * 0.050), vec![
            (dvec2(x1, 0.), dvec2(x2, 25.4)),
            (dvec2(x1, 0. + 215.), dvec2(x2, 25.4 + 215.)),
        ]));
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

        let x1 = x1.max(0.);
        let x4 = x4.min(1005.);

        raw_hats.push(((slopes_short[ii - 1] * 0.050, f64::NAN), vec![
            (dvec2(x1, 0.), dvec2(x2, 25.4)),
            (dvec2(x3, 0.), dvec2(x4, 25.4)),
        ]));
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

        let x1 = x1.max(0.);
        let x4 = x4.min(1240.);

        raw_hats.push(((slopes_long[ii - 1] * 0.050, f64::NAN), vec![
            (dvec2(x1, 0. + 215.), dvec2(x2, 25.4 + 215.)),
            (dvec2(x3, 0. + 215.), dvec2(x4, 25.4 + 215.)),
        ]));
    }

    let mut measurements = Vec::new();
    for ((angle_x, angle_y), rects) in raw_hats {
        let mut tris = Vec::new();
        for (bl, tr) in rects {
            tris.extend(Triangle::rectangle(bl, tr));
        }

        // Default angle bounds to use when there's an unspecified angle
        let default_angle = Bounds::Range(-0.05 * 10., 0.05 * 10.);

        // Default measurement uncertainty to apply to angles (mm/m)
        let uncertainty = 0.0;

        // Generate ranges for the X and Y slopes
        let angle_x = if angle_x.is_finite() {
            Bounds::Range(angle_x - uncertainty, angle_x + uncertainty)
        } else { default_angle };
        let angle_y = if angle_y.is_finite() {
            Bounds::Range(angle_y - uncertainty, angle_y + uncertainty)
        } else { default_angle };

        measurements.push(Measurement {
            contact:        tris,
            center_of_mass: None,
            plane:          Plane::default(),
            offset:         Bounds::Range(-0.25, 0.25),
            angle_x, angle_y,
        });
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

    struct State {
        /// Best planes for measurements we've found so far
        best_planes: Vec<Plane>,

        /// Lowest error score
        best: f64,

        /// Number of iterations
        iters: u64,
    }

    // Save the current state
    let state: [Arc<Mutex<State>>; 1] = std::array::from_fn(|_| {
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

    for state in state.iter() {
        let mut measurements = measurements.clone();
        let state = state.clone();
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
                for _ in 0..::rand::random::<usize>() % 4 + 1 {
                    let sel = ::rand::random::<usize>() % measurements.len();
                    measurements[sel].mutate();
                }

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
                let avg_dist = sum_dist / num_dist;

                {
                    let mut state = state.lock().unwrap();

                    if avg_dist < state.best {
                        state.best_planes.iter_mut().enumerate()
                            .for_each(|(ii, x)| *x = measurements[ii].plane);

                        state.best = avg_dist;
                    }
                }
            }
        });
    }

    let mut renderer = Renderer::default();
    let it = std::time::Instant::now();
    for frame in 1u64.. {
        use std::fmt::Write;

        // Render the best measurements we've had so far
        clear_background(DARKGRAY);

        let mut msg = String::new();
        let mut range = (0., 0.);
        for state in state.iter() {
            let score = {
                let state = state.lock().unwrap();

                // Restore best parameters
                measurements.iter_mut().enumerate()
                    .for_each(|(ii, x)| x.plane = state.best_planes[ii]);

                state.best
            };

            // Update samples
            renderer.generate_samples(&measurements, 100);
            renderer.simplify();

            // Average the data
            let mut sum = dvec3(0., 0., 0.);
            let mut cnt = 0.;
            for pt in renderer.samples.iter().map(|(pt, _, z)| pt.extend(*z)) {
                sum += pt;
                cnt += 1.;
            }
            let avg = sum / cnt;

            // Camera target is the center of mass
            let target = vec3(avg.x as f32, avg.y as f32, (avg.z * Z_SCALE) as f32);

            set_camera(&Camera3D {
                position: vec3(target.x, target.y - 600., target.z + 600.),
                up: vec3(0., 0., 1.),
                target,
                ..Default::default()
            });

            renderer.draw();

            writeln!(&mut msg, "Score {:7.3} um err/point | Range {:7.3} um",
                score * 1e3,
                (renderer.max_z - renderer.min_z) * 1e3).unwrap();
            range = (renderer.min_z, renderer.max_z - renderer.min_z);
        }

        writeln!(&mut msg,
            "FPS: {}", frame as f64 / it.elapsed().as_secs_f64()).unwrap();

        set_default_camera();
        draw_multiline_text(&msg, 0., 80., 24., Some(1.), BLACK);

        let width = screen_width() as usize;

        draw_rectangle(0., 5., width as f32, 30., BLACK);

        for pixel in 0..width {
            let normal = pixel as f64 / width as f64;
            draw_rectangle(pixel as f32, 10., 1., 20., Renderer::color(normal));

            if pixel % 100 == 0 {
                draw_text(&format!("{:.2} um", (normal * range.1 + range.0) * 1e3), pixel as f32, 50., 18., Renderer::color(normal));
            }
        }

        next_frame().await;
    }
}

