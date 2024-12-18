// This was taken before we were using CSV data

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
    for &(_, y) in data {
        assert!(y < 0.);
    }

    //let radius = 23.91 / 2.;
    let radius = 30.;
    let sides = 16;
    let circle = true;

    // Compute coords for data points
    for (ii, &(angle_x, angle_y)) in data.iter().enumerate() {
        // y_coord
        // ^
        // ^
        // ^
        // o > > > x_coord
        // origin
        let origin = dvec2(((ii % 8 + 1) * 50) as f64,
            ((ii / 8 + 1) * 50) as f64);

        let x_coord = origin + dvec2(150., 0.);
        let y_coord = origin + dvec2(0., 150.);

        // Generate ranges for the X and Y slopes
        let angle_x = Bounds::Constant(angle_x);
        let angle_y = Bounds::Constant(angle_y);

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
            last_vals: (f64::NAN, f64::NAN, f64::NAN),
            angle_x, angle_y,
        });
    }

    // Compute coords for special data points
    //
    // These are rotated clockwise 90 degrees
    for (ii, &(angle_x, angle_y)) in special_data.iter().enumerate() {
        // y_coord
        // ^
        // ^
        // ^
        // o > > > x_coord
        // origin
        let origin = dvec2(((ii / 3 + 6) * 50) as f64,
            ((ii % 3) as isize * -50) as f64 + 400.);

        let x_coord = origin + dvec2(150., 0.);
        let y_coord = origin - dvec2(0., 150.);

        let (angle_x, angle_y) = (angle_y, -angle_x);

        // Generate ranges for the X and Y slopes
        let angle_x = Bounds::Constant(angle_x);
        let angle_y = Bounds::Constant(angle_y);

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
            last_vals: (f64::NAN, f64::NAN, f64::NAN),
            angle_x, angle_y,
        });
    }

    // Compute average X angle and average Y angle
    let mut sumx = 0.;
    let mut numx = 0.;
    let mut sumy = 0.;
    let mut numy = 0.;
    for Measurement { angle_x, angle_y, .. } in measurements.iter() {
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

    for Measurement { angle_x, angle_y, .. } in measurements.iter_mut() {
        if let (Bounds::Constant(angle_x), Bounds::Constant(angle_y)) = (angle_x, angle_y) {
            *angle_x -= avgx;
            *angle_y -= avgy;
        }
    }

    //panic!();
}
