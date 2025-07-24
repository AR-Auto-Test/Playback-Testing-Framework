package org.opencv.imgproc;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;

/* loaded from: classes2.dex */
public class Moments {
    public double m00;
    public double m01;
    public double m02;
    public double m03;
    public double m10;
    public double m11;
    public double m12;
    public double m20;
    public double m21;
    public double m30;
    public double mu02;
    public double mu03;
    public double mu11;
    public double mu12;
    public double mu20;
    public double mu21;
    public double mu30;
    public double nu02;
    public double nu03;
    public double nu11;
    public double nu12;
    public double nu20;
    public double nu21;
    public double nu30;

    public Moments(double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9, double d10, double d11) {
        this.m00 = d2;
        this.m10 = d3;
        this.m01 = d4;
        this.m20 = d5;
        this.m11 = d6;
        this.m02 = d7;
        this.m30 = d8;
        this.m21 = d9;
        this.m12 = d10;
        this.m03 = d11;
        completeState();
    }

    public void completeState() {
        double d2;
        double d3;
        int i = (Math.abs(this.m00) > 1.0E-8d ? 1 : (Math.abs(this.m00) == 1.0E-8d ? 0 : -1));
        double d4 = ShadowDrawableWrapper.COS_45;
        if (i > 0) {
            double d5 = 1.0d / this.m00;
            d3 = this.m01 * d5;
            d4 = this.m10 * d5;
            d2 = d5;
        } else {
            d2 = 0.0d;
            d3 = 0.0d;
        }
        double d6 = this.m20;
        double d7 = this.m10;
        double d8 = d6 - (d7 * d4);
        double d9 = this.m11 - (d7 * d3);
        double d10 = this.m02;
        double d11 = d2;
        double d12 = this.m01;
        double d13 = d10 - (d12 * d3);
        this.mu20 = d8;
        this.mu11 = d9;
        this.mu02 = d13;
        this.mu30 = this.m30 - (((d4 * d7) + (d8 * 3.0d)) * d4);
        double d14 = d9 + d9;
        this.mu21 = (this.m21 - (((d4 * d12) + d14) * d4)) - (d8 * d3);
        this.mu12 = (this.m12 - (((d7 * d3) + d14) * d3)) - (d4 * d13);
        this.mu03 = this.m03 - (((d12 * d3) + (d13 * 3.0d)) * d3);
        double d15 = d11 * d11;
        double sqrt = Math.sqrt(Math.abs(d11)) * d15;
        this.nu20 = this.mu20 * d15;
        this.nu11 = this.mu11 * d15;
        this.nu02 = this.mu02 * d15;
        this.nu30 = this.mu30 * sqrt;
        this.nu21 = this.mu21 * sqrt;
        this.nu12 = this.mu12 * sqrt;
        this.nu03 = this.mu03 * sqrt;
    }

    public double get_m00() {
        return this.m00;
    }

    public double get_m01() {
        return this.m01;
    }

    public double get_m02() {
        return this.m02;
    }

    public double get_m03() {
        return this.m03;
    }

    public double get_m10() {
        return this.m10;
    }

    public double get_m11() {
        return this.m11;
    }

    public double get_m12() {
        return this.m12;
    }

    public double get_m20() {
        return this.m20;
    }

    public double get_m21() {
        return this.m21;
    }

    public double get_m30() {
        return this.m30;
    }

    public double get_mu02() {
        return this.mu02;
    }

    public double get_mu03() {
        return this.mu03;
    }

    public double get_mu11() {
        return this.mu11;
    }

    public double get_mu12() {
        return this.mu12;
    }

    public double get_mu20() {
        return this.mu20;
    }

    public double get_mu21() {
        return this.mu21;
    }

    public double get_mu30() {
        return this.mu30;
    }

    public double get_nu02() {
        return this.nu02;
    }

    public double get_nu03() {
        return this.nu03;
    }

    public double get_nu11() {
        return this.nu11;
    }

    public double get_nu12() {
        return this.nu12;
    }

    public double get_nu20() {
        return this.nu20;
    }

    public double get_nu21() {
        return this.nu21;
    }

    public double get_nu30() {
        return this.nu30;
    }

    public void set(double[] dArr) {
        double d2 = ShadowDrawableWrapper.COS_45;
        if (dArr != null) {
            this.m00 = dArr.length > 0 ? dArr[0] : 0.0d;
            this.m10 = dArr.length > 1 ? dArr[1] : 0.0d;
            this.m01 = dArr.length > 2 ? dArr[2] : 0.0d;
            this.m20 = dArr.length > 3 ? dArr[3] : 0.0d;
            this.m11 = dArr.length > 4 ? dArr[4] : 0.0d;
            this.m02 = dArr.length > 5 ? dArr[5] : 0.0d;
            this.m30 = dArr.length > 6 ? dArr[6] : 0.0d;
            this.m21 = dArr.length > 7 ? dArr[7] : 0.0d;
            this.m12 = dArr.length > 8 ? dArr[8] : 0.0d;
            if (dArr.length > 9) {
                d2 = dArr[9];
            }
            this.m03 = d2;
            completeState();
            return;
        }
        this.m00 = ShadowDrawableWrapper.COS_45;
        this.m10 = ShadowDrawableWrapper.COS_45;
        this.m01 = ShadowDrawableWrapper.COS_45;
        this.m20 = ShadowDrawableWrapper.COS_45;
        this.m11 = ShadowDrawableWrapper.COS_45;
        this.m02 = ShadowDrawableWrapper.COS_45;
        this.m30 = ShadowDrawableWrapper.COS_45;
        this.m21 = ShadowDrawableWrapper.COS_45;
        this.m12 = ShadowDrawableWrapper.COS_45;
        this.m03 = ShadowDrawableWrapper.COS_45;
        this.mu20 = ShadowDrawableWrapper.COS_45;
        this.mu11 = ShadowDrawableWrapper.COS_45;
        this.mu02 = ShadowDrawableWrapper.COS_45;
        this.mu30 = ShadowDrawableWrapper.COS_45;
        this.mu21 = ShadowDrawableWrapper.COS_45;
        this.mu12 = ShadowDrawableWrapper.COS_45;
        this.mu03 = ShadowDrawableWrapper.COS_45;
        this.nu20 = ShadowDrawableWrapper.COS_45;
        this.nu11 = ShadowDrawableWrapper.COS_45;
        this.nu02 = ShadowDrawableWrapper.COS_45;
        this.nu30 = ShadowDrawableWrapper.COS_45;
        this.nu21 = ShadowDrawableWrapper.COS_45;
        this.nu12 = ShadowDrawableWrapper.COS_45;
        this.nu03 = ShadowDrawableWrapper.COS_45;
    }

    public void set_m00(double d2) {
        this.m00 = d2;
    }

    public void set_m01(double d2) {
        this.m01 = d2;
    }

    public void set_m02(double d2) {
        this.m02 = d2;
    }

    public void set_m03(double d2) {
        this.m03 = d2;
    }

    public void set_m10(double d2) {
        this.m10 = d2;
    }

    public void set_m11(double d2) {
        this.m11 = d2;
    }

    public void set_m12(double d2) {
        this.m12 = d2;
    }

    public void set_m20(double d2) {
        this.m20 = d2;
    }

    public void set_m21(double d2) {
        this.m21 = d2;
    }

    public void set_m30(double d2) {
        this.m30 = d2;
    }

    public void set_mu02(double d2) {
        this.mu02 = d2;
    }

    public void set_mu03(double d2) {
        this.mu03 = d2;
    }

    public void set_mu11(double d2) {
        this.mu11 = d2;
    }

    public void set_mu12(double d2) {
        this.mu12 = d2;
    }

    public void set_mu20(double d2) {
        this.mu20 = d2;
    }

    public void set_mu21(double d2) {
        this.mu21 = d2;
    }

    public void set_mu30(double d2) {
        this.mu30 = d2;
    }

    public void set_nu02(double d2) {
        this.nu02 = d2;
    }

    public void set_nu03(double d2) {
        this.nu03 = d2;
    }

    public void set_nu11(double d2) {
        this.nu11 = d2;
    }

    public void set_nu12(double d2) {
        this.nu12 = d2;
    }

    public void set_nu20(double d2) {
        this.nu20 = d2;
    }

    public void set_nu21(double d2) {
        this.nu21 = d2;
    }

    public void set_nu30(double d2) {
        this.nu30 = d2;
    }

    public String toString() {
        StringBuilder x = a.x("Moments [ \nm00=");
        x.append(this.m00);
        x.append(", \nm10=");
        x.append(this.m10);
        x.append(", m01=");
        x.append(this.m01);
        x.append(", \nm20=");
        x.append(this.m20);
        x.append(", m11=");
        x.append(this.m11);
        x.append(", m02=");
        x.append(this.m02);
        x.append(", \nm30=");
        x.append(this.m30);
        x.append(", m21=");
        x.append(this.m21);
        x.append(", m12=");
        x.append(this.m12);
        x.append(", m03=");
        x.append(this.m03);
        x.append(", \nmu20=");
        x.append(this.mu20);
        x.append(", mu11=");
        x.append(this.mu11);
        x.append(", mu02=");
        x.append(this.mu02);
        x.append(", \nmu30=");
        x.append(this.mu30);
        x.append(", mu21=");
        x.append(this.mu21);
        x.append(", mu12=");
        x.append(this.mu12);
        x.append(", mu03=");
        x.append(this.mu03);
        x.append(", \nnu20=");
        x.append(this.nu20);
        x.append(", nu11=");
        x.append(this.nu11);
        x.append(", nu02=");
        x.append(this.nu02);
        x.append(", \nnu30=");
        x.append(this.nu30);
        x.append(", nu21=");
        x.append(this.nu21);
        x.append(", nu12=");
        x.append(this.nu12);
        x.append(", nu03=");
        x.append(this.nu03);
        x.append(", \n]");
        return x.toString();
    }

    public Moments() {
        this(ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45, ShadowDrawableWrapper.COS_45);
    }

    public Moments(double[] dArr) {
        set(dArr);
    }
}