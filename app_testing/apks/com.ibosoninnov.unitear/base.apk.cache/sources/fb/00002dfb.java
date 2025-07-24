package org.opencv.core;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import java.util.Arrays;

/* loaded from: classes2.dex */
public class Scalar {
    public double[] val;

    public Scalar(double d2, double d3, double d4, double d5) {
        this.val = new double[]{d2, d3, d4, d5};
    }

    public static Scalar all(double d2) {
        return new Scalar(d2, d2, d2, d2);
    }

    public Scalar conj() {
        double[] dArr = this.val;
        return new Scalar(dArr[0], -dArr[1], -dArr[2], -dArr[3]);
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        return (obj instanceof Scalar) && Arrays.equals(this.val, ((Scalar) obj).val);
    }

    public int hashCode() {
        return Arrays.hashCode(this.val) + 31;
    }

    public boolean isReal() {
        double[] dArr = this.val;
        return dArr[1] == ShadowDrawableWrapper.COS_45 && dArr[2] == ShadowDrawableWrapper.COS_45 && dArr[3] == ShadowDrawableWrapper.COS_45;
    }

    public Scalar mul(Scalar scalar, double d2) {
        double[] dArr = this.val;
        double d3 = dArr[0];
        double[] dArr2 = scalar.val;
        return new Scalar(d3 * dArr2[0] * d2, dArr[1] * dArr2[1] * d2, dArr[2] * dArr2[2] * d2, dArr[3] * dArr2[3] * d2);
    }

    public void set(double[] dArr) {
        double d2 = ShadowDrawableWrapper.COS_45;
        if (dArr != null) {
            double[] dArr2 = this.val;
            dArr2[0] = dArr.length > 0 ? dArr[0] : 0.0d;
            dArr2[1] = dArr.length > 1 ? dArr[1] : 0.0d;
            dArr2[2] = dArr.length > 2 ? dArr[2] : 0.0d;
            if (dArr.length > 3) {
                d2 = dArr[3];
            }
            dArr2[3] = d2;
            return;
        }
        double[] dArr3 = this.val;
        dArr3[3] = 0.0d;
        dArr3[2] = 0.0d;
        dArr3[1] = 0.0d;
        dArr3[0] = 0.0d;
    }

    public String toString() {
        StringBuilder x = a.x("[");
        x.append(this.val[0]);
        x.append(", ");
        x.append(this.val[1]);
        x.append(", ");
        x.append(this.val[2]);
        x.append(", ");
        x.append(this.val[3]);
        x.append("]");
        return x.toString();
    }

    /* JADX DEBUG: Method merged with bridge method */
    public Scalar clone() {
        return new Scalar(this.val);
    }

    public Scalar mul(Scalar scalar) {
        return mul(scalar, 1.0d);
    }

    public Scalar(double d2, double d3, double d4) {
        this.val = new double[]{d2, d3, d4, ShadowDrawableWrapper.COS_45};
    }

    public Scalar(double d2, double d3) {
        this.val = new double[]{d2, d3, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45};
    }

    public Scalar(double d2) {
        this.val = new double[]{d2, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45};
    }

    public Scalar(double[] dArr) {
        if (dArr != null && dArr.length == 4) {
            this.val = (double[]) dArr.clone();
            return;
        }
        this.val = new double[4];
        set(dArr);
    }
}