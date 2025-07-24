package org.opencv.core;

import com.google.android.material.shadow.ShadowDrawableWrapper;

/* loaded from: classes2.dex */
public class Size {
    public double height;
    public double width;

    public Size(double d2, double d3) {
        this.width = d2;
        this.height = d3;
    }

    public double area() {
        return this.width * this.height;
    }

    public boolean empty() {
        return this.width <= ShadowDrawableWrapper.COS_45 || this.height <= ShadowDrawableWrapper.COS_45;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof Size) {
            Size size = (Size) obj;
            return this.width == size.width && this.height == size.height;
        }
        return false;
    }

    public int hashCode() {
        long doubleToLongBits = Double.doubleToLongBits(this.height);
        long doubleToLongBits2 = Double.doubleToLongBits(this.width);
        return ((((int) (doubleToLongBits ^ (doubleToLongBits >>> 32))) + 31) * 31) + ((int) ((doubleToLongBits2 >>> 32) ^ doubleToLongBits2));
    }

    public void set(double[] dArr) {
        double d2 = ShadowDrawableWrapper.COS_45;
        if (dArr != null) {
            this.width = dArr.length > 0 ? dArr[0] : 0.0d;
            if (dArr.length > 1) {
                d2 = dArr[1];
            }
            this.height = d2;
            return;
        }
        this.width = ShadowDrawableWrapper.COS_45;
        this.height = ShadowDrawableWrapper.COS_45;
    }

    public String toString() {
        return ((int) this.width) + "x" + ((int) this.height);
    }

    /* JADX DEBUG: Method merged with bridge method */
    public Size clone() {
        return new Size(this.width, this.height);
    }

    public Size() {
        this(ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45);
    }

    public Size(Point point) {
        this.width = point.x;
        this.height = point.y;
    }

    public Size(double[] dArr) {
        set(dArr);
    }
}