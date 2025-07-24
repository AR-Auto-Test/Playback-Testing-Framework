package com.google.ar.sceneform.math;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Quaternion {
    private static final float SLERP_THRESHOLD = 0.9995f;
    public float w;
    public float x;
    public float y;
    public float z;

    public Quaternion() {
        this.x = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.y = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.z = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.w = 1.0f;
    }

    public static Quaternion add(Quaternion quaternion, Quaternion quaternion2) {
        Preconditions.checkNotNull(quaternion, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(quaternion2, "Parameter \"rhs\" was null.");
        Quaternion quaternion3 = new Quaternion();
        quaternion3.x = quaternion.x + quaternion2.x;
        quaternion3.y = quaternion.y + quaternion2.y;
        quaternion3.z = quaternion.z + quaternion2.z;
        quaternion3.w = quaternion.w + quaternion2.w;
        return quaternion3;
    }

    public static Quaternion axisAngle(Vector3 vector3, float f2) {
        Preconditions.checkNotNull(vector3, "Parameter \"axis\" was null.");
        Quaternion quaternion = new Quaternion();
        double radians = Math.toRadians(f2) / 2.0d;
        double sin = Math.sin(radians);
        quaternion.x = (float) (vector3.x * sin);
        quaternion.y = (float) (vector3.y * sin);
        quaternion.z = (float) (vector3.z * sin);
        quaternion.w = (float) Math.cos(radians);
        quaternion.normalize();
        return quaternion;
    }

    public static float dot(Quaternion quaternion, Quaternion quaternion2) {
        Preconditions.checkNotNull(quaternion, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(quaternion2, "Parameter \"rhs\" was null.");
        float f2 = (quaternion.y * quaternion2.y) + (quaternion.x * quaternion2.x);
        return (quaternion.w * quaternion2.w) + (quaternion.z * quaternion2.z) + f2;
    }

    public static boolean equals(Quaternion quaternion, Quaternion quaternion2) {
        Preconditions.checkNotNull(quaternion, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(quaternion2, "Parameter \"rhs\" was null.");
        return MathHelper.almostEqualRelativeAndAbs(dot(quaternion, quaternion2), 1.0f);
    }

    public static Quaternion eulerAngles(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"eulerAngles\" was null.");
        Quaternion quaternion = new Quaternion(Vector3.right(), vector3.x);
        Quaternion quaternion2 = new Quaternion(Vector3.up(), vector3.y);
        return multiply(multiply(quaternion2, quaternion), new Quaternion(Vector3.back(), vector3.z));
    }

    public static Quaternion identity() {
        return new Quaternion();
    }

    public static Vector3 inverseRotateVector(Quaternion quaternion, Vector3 vector3) {
        Preconditions.checkNotNull(quaternion, "Parameter \"q\" was null.");
        Preconditions.checkNotNull(vector3, "Parameter \"src\" was null.");
        Vector3 vector32 = new Vector3();
        float f2 = quaternion.w;
        float f3 = f2 * f2;
        float f4 = quaternion.x;
        float f5 = (-f4) * (-f4);
        float f6 = quaternion.y;
        float f7 = (-f6) * (-f6);
        float f8 = quaternion.z;
        float f9 = (-f8) * (-f8);
        float f10 = (-f8) * f2;
        float f11 = (-f4) * (-f6);
        float f12 = (-f4) * (-f8);
        float f13 = (-f6) * f2;
        float f14 = (-f6) * (-f8);
        float f15 = (-f4) * f2;
        float f16 = f11 + f10 + f10 + f11;
        float f17 = (((-f10) + f11) - f10) + f11;
        float f18 = f14 + f14;
        float f19 = f18 + f15 + f15;
        float f20 = (f18 - f15) - f15;
        float f21 = ((f9 - f7) - f5) + f3;
        float f22 = vector3.x;
        float f23 = vector3.y;
        float f24 = vector3.z;
        float f25 = (f13 + f12 + f12 + f13) * f24;
        vector32.x = f25 + (f17 * f23) + ((((f3 + f5) - f9) - f7) * f22);
        float f26 = f20 * f24;
        vector32.y = f26 + ((((f7 - f9) + f3) - f5) * f23) + (f16 * f22);
        float f27 = f21 * f24;
        vector32.z = f27 + (f19 * f23) + ((((f12 - f13) + f12) - f13) * f22);
        return vector32;
    }

    public static Quaternion lerp(Quaternion quaternion, Quaternion quaternion2, float f2) {
        Preconditions.checkNotNull(quaternion, "Parameter \"a\" was null.");
        Preconditions.checkNotNull(quaternion2, "Parameter \"b\" was null.");
        return new Quaternion(MathHelper.lerp(quaternion.x, quaternion2.x, f2), MathHelper.lerp(quaternion.y, quaternion2.y, f2), MathHelper.lerp(quaternion.z, quaternion2.z, f2), MathHelper.lerp(quaternion.w, quaternion2.w, f2));
    }

    public static Quaternion lookRotation(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"forwardInWorld\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"desiredUpInWorld\" was null.");
        Quaternion rotationBetweenVectors = rotationBetweenVectors(Vector3.forward(), vector3);
        return multiply(rotationBetweenVectors(rotateVector(rotationBetweenVectors, Vector3.up()), Vector3.cross(Vector3.cross(vector3, vector32), vector3)), rotationBetweenVectors);
    }

    public static Quaternion multiply(Quaternion quaternion, Quaternion quaternion2) {
        Preconditions.checkNotNull(quaternion, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(quaternion2, "Parameter \"rhs\" was null.");
        float f2 = quaternion.x;
        float f3 = quaternion.y;
        float f4 = quaternion.z;
        float f5 = quaternion.w;
        float f6 = quaternion2.x;
        float f7 = quaternion2.y;
        float f8 = quaternion2.z;
        float f9 = quaternion2.w;
        return new Quaternion(((f3 * f8) + ((f2 * f9) + (f5 * f6))) - (f4 * f7), (f4 * f6) + (f3 * f9) + ((f5 * f7) - (f2 * f8)), (f4 * f9) + (((f2 * f7) + (f5 * f8)) - (f3 * f6)), (((f5 * f9) - (f2 * f6)) - (f3 * f7)) - (f4 * f8));
    }

    public static Vector3 rotateVector(Quaternion quaternion, Vector3 vector3) {
        Preconditions.checkNotNull(quaternion, "Parameter \"q\" was null.");
        Preconditions.checkNotNull(vector3, "Parameter \"src\" was null.");
        Vector3 vector32 = new Vector3();
        float f2 = quaternion.w;
        float f3 = f2 * f2;
        float f4 = quaternion.x;
        float f5 = f4 * f4;
        float f6 = quaternion.y;
        float f7 = f6 * f6;
        float f8 = quaternion.z;
        float f9 = f8 * f8;
        float f10 = f8 * f2;
        float f11 = f4 * f6;
        float f12 = f4 * f8;
        float f13 = f6 * f2;
        float f14 = f6 * f8;
        float f15 = f4 * f2;
        float f16 = f11 + f10 + f10 + f11;
        float f17 = (((-f10) + f11) - f10) + f11;
        float f18 = f14 + f14;
        float f19 = f18 + f15 + f15;
        float f20 = (f18 - f15) - f15;
        float f21 = ((f9 - f7) - f5) + f3;
        float f22 = vector3.x;
        float f23 = vector3.y;
        float f24 = vector3.z;
        float f25 = (f13 + f12 + f12 + f13) * f24;
        vector32.x = f25 + (f17 * f23) + ((((f3 + f5) - f9) - f7) * f22);
        float f26 = f20 * f24;
        vector32.y = f26 + ((((f7 - f9) + f3) - f5) * f23) + (f16 * f22);
        float f27 = f21 * f24;
        vector32.z = f27 + (f19 * f23) + ((((f12 - f13) + f12) - f13) * f22);
        return vector32;
    }

    public static Quaternion rotationBetweenVectors(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"start\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"end\" was null.");
        Vector3 normalized = vector3.normalized();
        Vector3 normalized2 = vector32.normalized();
        float dot = Vector3.dot(normalized, normalized2);
        if (dot < -0.999f) {
            Vector3 cross = Vector3.cross(Vector3.back(), normalized);
            if (cross.lengthSquared() < 0.01f) {
                cross = Vector3.cross(Vector3.right(), normalized);
            }
            return axisAngle(cross.normalized(), 180.0f);
        }
        Vector3 cross2 = Vector3.cross(normalized, normalized2);
        float sqrt = (float) Math.sqrt((dot + 1.0d) * 2.0d);
        float f2 = 1.0f / sqrt;
        return new Quaternion(cross2.x * f2, cross2.y * f2, cross2.z * f2, sqrt * 0.5f);
    }

    public static Quaternion slerp(Quaternion quaternion, Quaternion quaternion2, float f2) {
        Preconditions.checkNotNull(quaternion, "Parameter \"start\" was null.");
        Preconditions.checkNotNull(quaternion2, "Parameter \"end\" was null.");
        Quaternion normalized = quaternion.normalized();
        Quaternion normalized2 = quaternion2.normalized();
        double dot = dot(normalized, normalized2);
        if (dot < ShadowDrawableWrapper.COS_45) {
            normalized2 = normalized2.negated();
            dot = -dot;
        }
        if (dot > 0.9994999766349792d) {
            return lerp(normalized, normalized2, f2);
        }
        double max = Math.max(-1.0d, Math.min(1.0d, dot));
        double acos = Math.acos(max);
        double d2 = f2 * acos;
        return add(normalized.scaled((float) (Math.cos(d2) - ((Math.sin(d2) * max) / Math.sin(acos)))), normalized2.scaled((float) (Math.sin(d2) / Math.sin(acos)))).normalized();
    }

    public int hashCode() {
        int floatToIntBits = Float.floatToIntBits(this.x);
        int floatToIntBits2 = Float.floatToIntBits(this.y);
        return Float.floatToIntBits(this.z) + ((floatToIntBits2 + ((floatToIntBits + ((Float.floatToIntBits(this.w) + 31) * 31)) * 31)) * 31);
    }

    public Quaternion inverted() {
        return new Quaternion(-this.x, -this.y, -this.z, this.w);
    }

    public Quaternion negated() {
        return new Quaternion(-this.x, -this.y, -this.z, -this.w);
    }

    public boolean normalize() {
        float dot = dot(this, this);
        if (MathHelper.almostEqualRelativeAndAbs(dot, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            setIdentity();
            return false;
        } else if (dot != 1.0f) {
            float sqrt = (float) (1.0d / Math.sqrt(dot));
            this.x *= sqrt;
            this.y *= sqrt;
            this.z *= sqrt;
            this.w *= sqrt;
            return true;
        } else {
            return true;
        }
    }

    public Quaternion normalized() {
        Quaternion quaternion = new Quaternion(this);
        quaternion.normalize();
        return quaternion;
    }

    public Quaternion scaled(float f2) {
        Quaternion quaternion = new Quaternion();
        quaternion.x = this.x * f2;
        quaternion.y = this.y * f2;
        quaternion.z = this.z * f2;
        quaternion.w = this.w * f2;
        return quaternion;
    }

    public void set(Quaternion quaternion) {
        Preconditions.checkNotNull(quaternion, "Parameter \"q\" was null.");
        this.x = quaternion.x;
        this.y = quaternion.y;
        this.z = quaternion.z;
        this.w = quaternion.w;
        normalize();
    }

    public void setIdentity() {
        this.x = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.y = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.z = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.w = 1.0f;
    }

    public String toString() {
        StringBuilder x = a.x("[x=");
        x.append(this.x);
        x.append(", y=");
        x.append(this.y);
        x.append(", z=");
        x.append(this.z);
        x.append(", w=");
        x.append(this.w);
        x.append("]");
        return x.toString();
    }

    public boolean equals(Object obj) {
        if (obj instanceof Quaternion) {
            if (this == obj) {
                return true;
            }
            return equals(this, (Quaternion) obj);
        }
        return false;
    }

    public Quaternion(float f2, float f3, float f4, float f5) {
        set(f2, f3, f4, f5);
    }

    public void set(Vector3 vector3, float f2) {
        Preconditions.checkNotNull(vector3, "Parameter \"axis\" was null.");
        set(axisAngle(vector3, f2));
    }

    public Quaternion(Quaternion quaternion) {
        Preconditions.checkNotNull(quaternion, "Parameter \"q\" was null.");
        set(quaternion);
    }

    public void set(float f2, float f3, float f4, float f5) {
        this.x = f2;
        this.y = f3;
        this.z = f4;
        this.w = f5;
        normalize();
    }

    public Quaternion(Vector3 vector3, float f2) {
        Preconditions.checkNotNull(vector3, "Parameter \"axis\" was null.");
        set(axisAngle(vector3, f2));
    }

    public Quaternion(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"eulerAngles\" was null.");
        set(eulerAngles(vector3));
    }
}