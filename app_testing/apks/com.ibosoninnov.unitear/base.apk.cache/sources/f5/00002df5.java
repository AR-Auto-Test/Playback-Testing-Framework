package org.opencv.core;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;

/* loaded from: classes2.dex */
public class Point {
    public double x;
    public double y;

    public Point(double d2, double d3) {
        this.x = d2;
        this.y = d3;
    }

    public double dot(Point point) {
        return (this.y * point.y) + (this.x * point.x);
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof Point) {
            Point point = (Point) obj;
            return this.x == point.x && this.y == point.y;
        }
        return false;
    }

    public int hashCode() {
        long doubleToLongBits = Double.doubleToLongBits(this.x);
        long doubleToLongBits2 = Double.doubleToLongBits(this.y);
        return ((((int) (doubleToLongBits ^ (doubleToLongBits >>> 32))) + 31) * 31) + ((int) ((doubleToLongBits2 >>> 32) ^ doubleToLongBits2));
    }

    public boolean inside(Rect rect) {
        return rect.contains(this);
    }

    public void set(double[] dArr) {
        double d2 = ShadowDrawableWrapper.COS_45;
        if (dArr != null) {
            this.x = dArr.length > 0 ? dArr[0] : 0.0d;
            if (dArr.length > 1) {
                d2 = dArr[1];
            }
            this.y = d2;
            return;
        }
        this.x = ShadowDrawableWrapper.COS_45;
        this.y = ShadowDrawableWrapper.COS_45;
    }

    public String toString() {
        StringBuilder x = a.x("{");
        x.append(this.x);
        x.append(", ");
        x.append(this.y);
        x.append("}");
        return x.toString();
    }

    /* JADX DEBUG: Method merged with bridge method */
    public Point clone() {
        return new Point(this.x, this.y);
    }

    public Point() {
        this(ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45);
    }

    public Point(double[] dArr) {
        this();
        set(dArr);
    }
}