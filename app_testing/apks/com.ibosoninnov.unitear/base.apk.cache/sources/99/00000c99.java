package com.google.android.filament;

/* loaded from: classes.dex */
public class Box {
    private final float[] mCenter;
    private final float[] mHalfExtent;

    public Box() {
        this.mCenter = new float[3];
        this.mHalfExtent = new float[3];
    }

    public float[] getCenter() {
        return this.mCenter;
    }

    public float[] getHalfExtent() {
        return this.mHalfExtent;
    }

    public void setCenter(float f2, float f3, float f4) {
        float[] fArr = this.mCenter;
        fArr[0] = f2;
        fArr[1] = f3;
        fArr[2] = f4;
    }

    public void setHalfExtent(float f2, float f3, float f4) {
        float[] fArr = this.mHalfExtent;
        fArr[0] = f2;
        fArr[1] = f3;
        fArr[2] = f4;
    }

    public Box(float f2, float f3, float f4, float f5, float f6, float f7) {
        this.mCenter = r1;
        this.mHalfExtent = r0;
        float[] fArr = {f2, f3, f4};
        float[] fArr2 = {f5, f6, f7};
    }

    public Box(float[] fArr, float[] fArr2) {
        this.mCenter = r1;
        this.mHalfExtent = r0;
        float[] fArr3 = {fArr[0], fArr[1], fArr[2]};
        float[] fArr4 = {fArr2[0], fArr2[1], fArr2[2]};
    }
}