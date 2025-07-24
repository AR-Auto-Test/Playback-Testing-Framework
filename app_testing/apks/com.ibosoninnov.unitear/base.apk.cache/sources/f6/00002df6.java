package org.opencv.core;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;

/* loaded from: classes2.dex */
public class Point3 {
    public double x;
    public double y;
    public double z;

    public Point3(double d2, double d3, double d4) {
        this.x = d2;
        this.y = d3;
        this.z = d4;
    }

    public Point3 cross(Point3 point3) {
        double d2 = this.y;
        double d3 = point3.z;
        double d4 = this.z;
        double d5 = point3.y;
        double d6 = (d2 * d3) - (d4 * d5);
        double d7 = point3.x;
        double d8 = this.x;
        return new Point3(d6, (d4 * d7) - (d3 * d8), (d8 * d5) - (d2 * d7));
    }

    public double dot(Point3 point3) {
        return (this.z * point3.z) + (this.y * point3.y) + (this.x * point3.x);
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof Point3) {
            Point3 point3 = (Point3) obj;
            return this.x == point3.x && this.y == point3.y && this.z == point3.z;
        }
        return false;
    }

    public int hashCode() {
        long doubleToLongBits = Double.doubleToLongBits(this.x);
        long doubleToLongBits2 = Double.doubleToLongBits(this.y);
        int i = ((((int) (doubleToLongBits ^ (doubleToLongBits >>> 32))) + 31) * 31) + ((int) (doubleToLongBits2 ^ (doubleToLongBits2 >>> 32)));
        long doubleToLongBits3 = Double.doubleToLongBits(this.z);
        return (i * 31) + ((int) ((doubleToLongBits3 >>> 32) ^ doubleToLongBits3));
    }

    public void set(double[] dArr) {
        double d2 = ShadowDrawableWrapper.COS_45;
        if (dArr != null) {
            this.x = dArr.length > 0 ? dArr[0] : 0.0d;
            this.y = dArr.length > 1 ? dArr[1] : 0.0d;
            if (dArr.length > 2) {
                d2 = dArr[2];
            }
            this.z = d2;
            return;
        }
        this.x = ShadowDrawableWrapper.COS_45;
        this.y = ShadowDrawableWrapper.COS_45;
        this.z = ShadowDrawableWrapper.COS_45;
    }

    public String toString() {
        StringBuilder x = a.x("{");
        x.append(this.x);
        x.append(", ");
        x.append(this.y);
        x.append(", ");
        x.append(this.z);
        x.append("}");
        return x.toString();
    }

    /* JADX DEBUG: Method merged with bridge method */
    public Point3 clone() {
        return new Point3(this.x, this.y, this.z);
    }

    public Point3() {
        this(ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45);
    }

    public Point3(Point point) {
        this.x = point.x;
        this.y = point.y;
        this.z = ShadowDrawableWrapper.COS_45;
    }

    public Point3(double[] dArr) {
        this();
        set(dArr);
    }
}