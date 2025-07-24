package org.opencv.core;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;

/* loaded from: classes2.dex */
public class TermCriteria {
    public static final int COUNT = 1;
    public static final int EPS = 2;
    public static final int MAX_ITER = 1;
    public double epsilon;
    public int maxCount;
    public int type;

    public TermCriteria(int i, int i2, double d2) {
        this.type = i;
        this.maxCount = i2;
        this.epsilon = d2;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof TermCriteria) {
            TermCriteria termCriteria = (TermCriteria) obj;
            return this.type == termCriteria.type && this.maxCount == termCriteria.maxCount && this.epsilon == termCriteria.epsilon;
        }
        return false;
    }

    public int hashCode() {
        long doubleToLongBits = Double.doubleToLongBits(this.type);
        long doubleToLongBits2 = Double.doubleToLongBits(this.maxCount);
        int i = ((((int) (doubleToLongBits ^ (doubleToLongBits >>> 32))) + 31) * 31) + ((int) (doubleToLongBits2 ^ (doubleToLongBits2 >>> 32)));
        long doubleToLongBits3 = Double.doubleToLongBits(this.epsilon);
        return (i * 31) + ((int) ((doubleToLongBits3 >>> 32) ^ doubleToLongBits3));
    }

    public void set(double[] dArr) {
        double d2 = ShadowDrawableWrapper.COS_45;
        if (dArr != null) {
            this.type = dArr.length > 0 ? (int) dArr[0] : 0;
            this.maxCount = dArr.length > 1 ? (int) dArr[1] : 0;
            if (dArr.length > 2) {
                d2 = dArr[2];
            }
            this.epsilon = d2;
            return;
        }
        this.type = 0;
        this.maxCount = 0;
        this.epsilon = ShadowDrawableWrapper.COS_45;
    }

    public String toString() {
        StringBuilder x = a.x("{ type: ");
        x.append(this.type);
        x.append(", maxCount: ");
        x.append(this.maxCount);
        x.append(", epsilon: ");
        x.append(this.epsilon);
        x.append("}");
        return x.toString();
    }

    /* JADX DEBUG: Method merged with bridge method */
    public TermCriteria clone() {
        return new TermCriteria(this.type, this.maxCount, this.epsilon);
    }

    public TermCriteria() {
        this(0, 0, ShadowDrawableWrapper.COS_45);
    }

    public TermCriteria(double[] dArr) {
        set(dArr);
    }
}