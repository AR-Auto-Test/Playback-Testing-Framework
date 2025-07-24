package org.opencv.core;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes2.dex */
public class KeyPoint {
    public float angle;
    public int class_id;
    public int octave;
    public Point pt;
    public float response;
    public float size;

    public KeyPoint(float f2, float f3, float f4, float f5, float f6, int i, int i2) {
        this.pt = new Point(f2, f3);
        this.size = f4;
        this.angle = f5;
        this.response = f6;
        this.octave = i;
        this.class_id = i2;
    }

    public String toString() {
        StringBuilder x = a.x("KeyPoint [pt=");
        x.append(this.pt);
        x.append(", size=");
        x.append(this.size);
        x.append(", angle=");
        x.append(this.angle);
        x.append(", response=");
        x.append(this.response);
        x.append(", octave=");
        x.append(this.octave);
        x.append(", class_id=");
        return a.s(x, this.class_id, "]");
    }

    public KeyPoint() {
        this(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0, -1);
    }

    public KeyPoint(float f2, float f3, float f4, float f5, float f6, int i) {
        this(f2, f3, f4, f5, f6, i, -1);
    }

    public KeyPoint(float f2, float f3, float f4, float f5, float f6) {
        this(f2, f3, f4, f5, f6, 0, -1);
    }

    public KeyPoint(float f2, float f3, float f4, float f5) {
        this(f2, f3, f4, f5, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0, -1);
    }

    public KeyPoint(float f2, float f3, float f4) {
        this(f2, f3, f4, -1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0, -1);
    }
}