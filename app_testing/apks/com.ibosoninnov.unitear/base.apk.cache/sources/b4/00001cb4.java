package com.google.ar.sceneform.math;

import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Matrix {
    public static final float[] IDENTITY_DATA = {1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f};
    private static final String TAG = "Matrix";
    public float[] data;

    public Matrix() {
        this.data = new float[16];
        set(IDENTITY_DATA);
    }

    public static boolean equals(Matrix matrix, Matrix matrix2) {
        Preconditions.checkNotNull(matrix, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(matrix2, "Parameter \"rhs\" was null.");
        boolean z = true;
        for (int i = 0; i < 16; i++) {
            z &= MathHelper.almostEqualRelativeAndAbs(matrix.data[i], matrix2.data[i]);
        }
        return z;
    }

    public static boolean invert(Matrix matrix, Matrix matrix2) {
        Preconditions.checkNotNull(matrix, "Parameter \"matrix\" was null.");
        Preconditions.checkNotNull(matrix2, "Parameter \"dest\" was null.");
        float[] fArr = matrix.data;
        float f2 = fArr[0];
        float f3 = fArr[1];
        float f4 = fArr[2];
        float f5 = fArr[3];
        float f6 = fArr[4];
        float f7 = fArr[5];
        float f8 = fArr[6];
        float f9 = fArr[7];
        float f10 = fArr[8];
        float f11 = fArr[9];
        float f12 = fArr[10];
        float f13 = fArr[11];
        float f14 = fArr[12];
        float f15 = fArr[13];
        float f16 = fArr[14];
        float f17 = fArr[15];
        float[] fArr2 = matrix2.data;
        fArr2[0] = (((f15 * f8) * f13) + (((f11 * f9) * f16) + ((((f7 * f12) * f17) - ((f7 * f13) * f16)) - ((f11 * f8) * f17)))) - ((f15 * f9) * f12);
        float f18 = -f6;
        float f19 = f6 * f13;
        float f20 = f10 * f8;
        float f21 = (f20 * f17) + (f19 * f16) + (f18 * f12 * f17);
        float f22 = f10 * f9;
        float f23 = f14 * f8;
        float f24 = f14 * f9;
        fArr2[4] = (f24 * f12) + ((f21 - (f22 * f16)) - (f23 * f13));
        float f25 = ((f6 * f11) * f17) - (f19 * f15);
        float f26 = f10 * f7;
        float f27 = (f22 * f15) + (f25 - (f26 * f17));
        float f28 = f14 * f7;
        fArr2[8] = ((f28 * f13) + f27) - (f24 * f11);
        float f29 = f26 * f16;
        float f30 = f23 * f11;
        fArr2[12] = f30 + (((f29 + (((f6 * f12) * f15) + ((f18 * f11) * f16))) - (f20 * f15)) - (f28 * f12));
        float f31 = -f3;
        float f32 = f11 * f4;
        float f33 = (f32 * f17) + (f3 * f13 * f16) + (f31 * f12 * f17);
        float f34 = f11 * f5;
        float f35 = f15 * f4;
        float f36 = f15 * f5;
        fArr2[1] = (f36 * f12) + ((f33 - (f34 * f16)) - (f35 * f13));
        float f37 = f2 * f12;
        float f38 = f2 * f13;
        float f39 = f10 * f4;
        float f40 = f10 * f5;
        float f41 = (f40 * f16) + (((f37 * f17) - (f38 * f16)) - (f39 * f17));
        float f42 = f14 * f4;
        float f43 = (f42 * f13) + f41;
        float f44 = f14 * f5;
        fArr2[5] = f43 - (f44 * f12);
        float f45 = -f2;
        float f46 = f10 * f3;
        float f47 = f14 * f3;
        fArr2[9] = (f44 * f11) + ((((f46 * f17) + ((f38 * f15) + ((f45 * f11) * f17))) - (f40 * f15)) - (f47 * f13));
        fArr2[13] = ((f47 * f12) + ((f39 * f15) + ((((f2 * f11) * f16) - (f37 * f15)) - (f46 * f16)))) - (f42 * f11);
        float f48 = f3 * f9;
        float f49 = f7 * f4;
        float f50 = f7 * f5;
        float f51 = f35 * f9;
        fArr2[2] = (f51 + ((f50 * f16) + ((((f3 * f8) * f17) - (f48 * f16)) - (f49 * f17)))) - (f36 * f8);
        float f52 = f2 * f9;
        float f53 = f6 * f4;
        float f54 = (f53 * f17) + (f52 * f16) + (f45 * f8 * f17);
        float f55 = f6 * f5;
        fArr2[6] = (f44 * f8) + ((f54 - (f55 * f16)) - (f42 * f9));
        float f56 = f2 * f7;
        float f57 = f6 * f3;
        fArr2[10] = ((f47 * f9) + ((f55 * f15) + (((f56 * f17) - (f52 * f15)) - (f17 * f57)))) - (f44 * f7);
        float f58 = f45 * f7;
        float f59 = f2 * f8;
        float f60 = f16 * f57;
        float f61 = f42 * f7;
        fArr2[14] = f61 + (((f60 + ((f59 * f15) + (f58 * f16))) - (f15 * f53)) - (f47 * f8));
        float f62 = f49 * f13;
        float f63 = f34 * f8;
        fArr2[3] = f63 + (((f62 + ((f48 * f12) + ((f31 * f8) * f13))) - (f50 * f12)) - (f32 * f9));
        fArr2[7] = ((f39 * f9) + ((f55 * f12) + (((f59 * f13) - (f52 * f12)) - (f53 * f13)))) - (f40 * f8);
        float f64 = f13 * f57;
        float f65 = f40 * f7;
        fArr2[11] = f65 + (((f64 + ((f52 * f11) + (f58 * f13))) - (f55 * f11)) - (f9 * f46));
        float f66 = f46 * f8;
        fArr2[15] = (f66 + ((f53 * f11) + (((f56 * f12) - (f59 * f11)) - (f57 * f12)))) - (f39 * f7);
        float f67 = f3 * fArr2[4];
        float f68 = f4 * fArr2[8];
        float f69 = (f5 * fArr2[12]) + f68 + f67 + (f2 * fArr2[0]);
        if (f69 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return false;
        }
        float f70 = 1.0f / f69;
        for (int i = 0; i < 16; i++) {
            float[] fArr3 = matrix2.data;
            fArr3[i] = fArr3[i] * f70;
        }
        return true;
    }

    public static void multiply(Matrix matrix, Matrix matrix2, Matrix matrix3) {
        Matrix matrix4 = matrix;
        Preconditions.checkNotNull(matrix4, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(matrix2, "Parameter \"rhs\" was null.");
        float f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        float f3 = 0.0f;
        float f4 = 0.0f;
        float f5 = 0.0f;
        float f6 = 0.0f;
        float f7 = 0.0f;
        float f8 = 0.0f;
        float f9 = 0.0f;
        float f10 = 0.0f;
        float f11 = 0.0f;
        float f12 = 0.0f;
        float f13 = 0.0f;
        float f14 = 0.0f;
        float f15 = 0.0f;
        float f16 = 0.0f;
        float f17 = 0.0f;
        int i = 0;
        while (true) {
            float f18 = f14;
            if (i < 4) {
                float[] fArr = matrix4.data;
                int i2 = i * 4;
                float f19 = fArr[i2 + 0];
                float f20 = fArr[i2 + 1];
                float f21 = fArr[i2 + 2];
                float f22 = fArr[i2 + 3];
                float[] fArr2 = matrix2.data;
                float f23 = fArr2[i + 0];
                float f24 = fArr2[i + 4];
                float f25 = fArr2[i + 8];
                float f26 = fArr2[i + 12];
                f2 = (f19 * f23) + f2;
                f3 = (f20 * f23) + f3;
                f4 = (f21 * f23) + f4;
                f5 = (f23 * f22) + f5;
                f6 = (f19 * f24) + f6;
                f7 = (f20 * f24) + f7;
                f8 = (f21 * f24) + f8;
                f9 = (f24 * f22) + f9;
                f10 = (f19 * f25) + f10;
                f11 = (f20 * f25) + f11;
                f12 = (f21 * f25) + f12;
                f13 = (f25 * f22) + f13;
                f15 = (f20 * f26) + f15;
                f16 = (f21 * f26) + f16;
                f17 = (f22 * f26) + f17;
                i++;
                matrix4 = matrix;
                f14 = (f19 * f26) + f18;
            } else {
                float[] fArr3 = matrix3.data;
                fArr3[0] = f2;
                fArr3[1] = f3;
                fArr3[2] = f4;
                fArr3[3] = f5;
                fArr3[4] = f6;
                fArr3[5] = f7;
                fArr3[6] = f8;
                fArr3[7] = f9;
                fArr3[8] = f10;
                fArr3[9] = f11;
                fArr3[10] = f12;
                fArr3[11] = f13;
                fArr3[12] = f18;
                fArr3[13] = f15;
                fArr3[14] = f16;
                fArr3[15] = f17;
                return;
            }
        }
    }

    public void decomposeRotation(Vector3 vector3, Quaternion quaternion) {
        float[] fArr = this.data;
        float f2 = fArr[0];
        float f3 = fArr[1];
        float f4 = fArr[2];
        float f5 = fArr[3];
        float f6 = fArr[4];
        float f7 = fArr[5];
        float f8 = fArr[6];
        float f9 = fArr[7];
        float f10 = fArr[8];
        float f11 = fArr[9];
        float f12 = fArr[10];
        float f13 = fArr[11];
        float f14 = fArr[12];
        float f15 = fArr[13];
        float f16 = fArr[14];
        float f17 = fArr[15];
        decomposeRotation(vector3, this);
        extractQuaternion(quaternion);
        float[] fArr2 = this.data;
        fArr2[0] = f2;
        fArr2[1] = f3;
        fArr2[2] = f4;
        fArr2[3] = f5;
        fArr2[4] = f6;
        fArr2[5] = f7;
        fArr2[6] = f8;
        fArr2[7] = f9;
        fArr2[8] = f10;
        fArr2[9] = f11;
        fArr2[10] = f12;
        fArr2[11] = f13;
        fArr2[12] = f14;
        fArr2[13] = f15;
        fArr2[14] = f16;
        fArr2[15] = f17;
    }

    public void decomposeScale(Vector3 vector3) {
        float[] fArr = this.data;
        Vector3 vector32 = new Vector3(fArr[0], fArr[1], fArr[2]);
        vector3.x = vector32.length();
        float[] fArr2 = this.data;
        vector32.set(fArr2[4], fArr2[5], fArr2[6]);
        vector3.y = vector32.length();
        float[] fArr3 = this.data;
        vector32.set(fArr3[8], fArr3[9], fArr3[10]);
        vector3.z = vector32.length();
    }

    public void decomposeTranslation(Vector3 vector3) {
        float[] fArr = this.data;
        vector3.x = fArr[12];
        vector3.y = fArr[13];
        vector3.z = fArr[14];
    }

    public void extractQuaternion(Quaternion quaternion) {
        float[] fArr = this.data;
        float f2 = fArr[0] + fArr[5] + fArr[10];
        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            float sqrt = ((float) Math.sqrt(f2 + 1.0d)) * 2.0f;
            quaternion.w = 0.25f * sqrt;
            float[] fArr2 = this.data;
            quaternion.x = (fArr2[6] - fArr2[9]) / sqrt;
            quaternion.y = (fArr2[8] - fArr2[2]) / sqrt;
            quaternion.z = (fArr2[1] - fArr2[4]) / sqrt;
        } else if (fArr[0] > fArr[5] && fArr[0] > fArr[10]) {
            float sqrt2 = ((float) Math.sqrt(((fArr[0] + 1.0f) - fArr[5]) - fArr[10])) * 2.0f;
            float[] fArr3 = this.data;
            quaternion.w = (fArr3[6] - fArr3[9]) / sqrt2;
            quaternion.x = 0.25f * sqrt2;
            quaternion.y = (fArr3[4] + fArr3[1]) / sqrt2;
            quaternion.z = (fArr3[8] + fArr3[2]) / sqrt2;
        } else if (fArr[5] > fArr[10]) {
            float sqrt3 = ((float) Math.sqrt(((fArr[5] + 1.0f) - fArr[0]) - fArr[10])) * 2.0f;
            float[] fArr4 = this.data;
            quaternion.w = (fArr4[8] - fArr4[2]) / sqrt3;
            quaternion.x = (fArr4[4] + fArr4[1]) / sqrt3;
            quaternion.y = 0.25f * sqrt3;
            quaternion.z = (fArr4[9] + fArr4[6]) / sqrt3;
        } else {
            float sqrt4 = ((float) Math.sqrt(((fArr[10] + 1.0f) - fArr[0]) - fArr[5])) * 2.0f;
            float[] fArr5 = this.data;
            quaternion.w = (fArr5[1] - fArr5[4]) / sqrt4;
            quaternion.x = (fArr5[8] + fArr5[2]) / sqrt4;
            quaternion.y = (fArr5[9] + fArr5[6]) / sqrt4;
            quaternion.z = sqrt4 * 0.25f;
        }
        quaternion.normalize();
    }

    public void makeRotation(Quaternion quaternion) {
        Preconditions.checkNotNull(quaternion, "Parameter \"rotation\" was null.");
        set(IDENTITY_DATA);
        quaternion.normalize();
        float f2 = quaternion.x;
        float f3 = f2 * f2;
        float f4 = quaternion.y;
        float f5 = f2 * f4;
        float f6 = quaternion.z;
        float f7 = f2 * f6;
        float f8 = quaternion.w;
        float f9 = f2 * f8;
        float f10 = f4 * f4;
        float f11 = f4 * f6;
        float f12 = f4 * f8;
        float f13 = f6 * f6;
        float f14 = f6 * f8;
        float[] fArr = this.data;
        fArr[0] = 1.0f - ((f10 + f13) * 2.0f);
        fArr[4] = (f5 - f14) * 2.0f;
        fArr[8] = (f7 + f12) * 2.0f;
        fArr[1] = (f5 + f14) * 2.0f;
        fArr[5] = 1.0f - ((f13 + f3) * 2.0f);
        fArr[9] = (f11 - f9) * 2.0f;
        fArr[2] = (f7 - f12) * 2.0f;
        fArr[6] = (f11 + f9) * 2.0f;
        fArr[10] = 1.0f - ((f3 + f10) * 2.0f);
    }

    public void makeScale(float f2) {
        Preconditions.checkNotNull(Float.valueOf(f2), "Parameter \"scale\" was null.");
        set(IDENTITY_DATA);
        float[] fArr = this.data;
        fArr[0] = f2;
        fArr[5] = f2;
        fArr[10] = f2;
    }

    public void makeTranslation(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"translation\" was null.");
        set(IDENTITY_DATA);
        setTranslation(vector3);
    }

    public void makeTrs(Vector3 vector3, Quaternion quaternion, Vector3 vector32) {
        float f2 = quaternion.x;
        float f3 = 1.0f - ((f2 * 2.0f) * f2);
        float f4 = quaternion.y;
        float f5 = f4 * f4;
        float f6 = quaternion.z;
        float f7 = f6 * 2.0f * f6;
        float f8 = f2 * 2.0f * f6;
        float f9 = quaternion.w;
        float f10 = f4 * 2.0f * f9;
        float f11 = f2 * 2.0f * f4;
        float f12 = f6 * 2.0f * f9;
        float f13 = f2 * 2.0f * f9;
        float f14 = f4 * 2.0f * f6;
        float[] fArr = this.data;
        float f15 = f5 * 2.0f;
        float f16 = vector32.x;
        fArr[0] = ((1.0f - f15) - f7) * f16;
        float f17 = vector32.y;
        fArr[4] = (f11 - f12) * f17;
        float f18 = vector32.z;
        fArr[8] = (f8 + f10) * f18;
        fArr[1] = (f11 + f12) * f16;
        fArr[5] = (f3 - f7) * f17;
        fArr[9] = (f14 - f13) * f18;
        fArr[2] = (f8 - f10) * f16;
        fArr[6] = (f14 + f13) * f17;
        fArr[10] = (f3 - f15) * f18;
        fArr[12] = vector3.x;
        fArr[13] = vector3.y;
        fArr[14] = vector3.z;
        fArr[15] = 1.0f;
    }

    public void set(float[] fArr) {
        if (fArr != null && fArr.length == 16) {
            for (int i = 0; i < fArr.length; i++) {
                this.data[i] = fArr[i];
            }
            return;
        }
        Log.w(TAG, "Cannot set Matrix, invalid data.");
    }

    public void setTranslation(Vector3 vector3) {
        float[] fArr = this.data;
        fArr[12] = vector3.x;
        fArr[13] = vector3.y;
        fArr[14] = vector3.z;
    }

    public Vector3 transformDirection(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"vector\" was null.");
        Vector3 vector32 = new Vector3();
        float f2 = vector3.x;
        float f3 = vector3.y;
        float f4 = vector3.z;
        float[] fArr = this.data;
        float f5 = fArr[0] * f2;
        vector32.x = f5;
        float f6 = (fArr[4] * f3) + f5;
        vector32.x = f6;
        vector32.x = (fArr[8] * f4) + f6;
        float f7 = fArr[1] * f2;
        vector32.y = f7;
        float f8 = (fArr[5] * f3) + f7;
        vector32.y = f8;
        vector32.y = (fArr[9] * f4) + f8;
        float f9 = fArr[2] * f2;
        vector32.z = f9;
        float f10 = (fArr[6] * f3) + f9;
        vector32.z = f10;
        vector32.z = (fArr[10] * f4) + f10;
        return vector32;
    }

    public Vector3 transformPoint(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"vector\" was null.");
        Vector3 vector32 = new Vector3();
        float f2 = vector3.x;
        float f3 = vector3.y;
        float f4 = vector3.z;
        float[] fArr = this.data;
        float f5 = fArr[0] * f2;
        vector32.x = f5;
        float f6 = (fArr[4] * f3) + f5;
        vector32.x = f6;
        float f7 = (fArr[8] * f4) + f6;
        vector32.x = f7;
        vector32.x = f7 + fArr[12];
        float f8 = fArr[1] * f2;
        vector32.y = f8;
        float f9 = (fArr[5] * f3) + f8;
        vector32.y = f9;
        float f10 = (fArr[9] * f4) + f9;
        vector32.y = f10;
        vector32.y = f10 + fArr[13];
        float f11 = fArr[2] * f2;
        vector32.z = f11;
        float f12 = (fArr[6] * f3) + f11;
        vector32.z = f12;
        float f13 = (fArr[10] * f4) + f12;
        vector32.z = f13;
        vector32.z = f13 + fArr[14];
        return vector32;
    }

    public Matrix(float[] fArr) {
        this.data = new float[16];
        set(fArr);
    }

    public void set(Matrix matrix) {
        Preconditions.checkNotNull(matrix, "Parameter \"m\" was null.");
        set(matrix.data);
    }

    public void makeScale(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"scale\" was null.");
        set(IDENTITY_DATA);
        float[] fArr = this.data;
        fArr[0] = vector3.x;
        fArr[5] = vector3.y;
        fArr[10] = vector3.z;
    }

    public void decomposeRotation(Vector3 vector3, Matrix matrix) {
        if (vector3.x != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            for (int i = 0; i < 3; i++) {
                matrix.data[i] = this.data[i] / vector3.x;
            }
        }
        matrix.data[3] = 0.0f;
        if (vector3.y != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            for (int i2 = 4; i2 < 7; i2++) {
                matrix.data[i2] = this.data[i2] / vector3.y;
            }
        }
        matrix.data[7] = 0.0f;
        if (vector3.z != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            for (int i3 = 8; i3 < 11; i3++) {
                matrix.data[i3] = this.data[i3] / vector3.z;
            }
        }
        float[] fArr = matrix.data;
        fArr[11] = 0.0f;
        fArr[12] = 0.0f;
        fArr[13] = 0.0f;
        fArr[14] = 0.0f;
        fArr[15] = 1.0f;
    }
}