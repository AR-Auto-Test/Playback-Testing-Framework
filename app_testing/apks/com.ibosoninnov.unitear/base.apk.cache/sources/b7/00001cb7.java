package com.google.ar.sceneform.math;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Vector3 {
    public float x;
    public float y;
    public float z;

    public Vector3() {
        this.x = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.y = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.z = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public static Vector3 add(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"rhs\" was null.");
        return new Vector3(vector3.x + vector32.x, vector3.y + vector32.y, vector3.z + vector32.z);
    }

    public static float angleBetweenVectors(Vector3 vector3, Vector3 vector32) {
        float length = vector32.length() * vector3.length();
        return MathHelper.almostEqualRelativeAndAbs(length, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) ? StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : (float) Math.toDegrees((float) Math.acos(MathHelper.clamp(dot(vector3, vector32) / length, -1.0f, 1.0f)));
    }

    public static Vector3 back() {
        Vector3 vector3 = new Vector3();
        vector3.setBack();
        return vector3;
    }

    public static float componentMax(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"a\" was null.");
        return Math.max(Math.max(vector3.x, vector3.y), vector3.z);
    }

    public static float componentMin(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"a\" was null.");
        return Math.min(Math.min(vector3.x, vector3.y), vector3.z);
    }

    public static Vector3 cross(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"rhs\" was null.");
        float f2 = vector3.x;
        float f3 = vector3.y;
        float f4 = vector3.z;
        float f5 = vector32.x;
        float f6 = vector32.y;
        float f7 = vector32.z;
        return new Vector3((f3 * f7) - (f4 * f6), (f4 * f5) - (f7 * f2), (f2 * f6) - (f3 * f5));
    }

    public static float dot(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"rhs\" was null.");
        float f2 = vector3.x * vector32.x;
        return (vector3.z * vector32.z) + (vector3.y * vector32.y) + f2;
    }

    public static Vector3 down() {
        Vector3 vector3 = new Vector3();
        vector3.setDown();
        return vector3;
    }

    public static boolean equals(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"rhs\" was null.");
        return MathHelper.almostEqualRelativeAndAbs(vector3.z, vector32.z) & MathHelper.almostEqualRelativeAndAbs(vector3.x, vector32.x) & true & MathHelper.almostEqualRelativeAndAbs(vector3.y, vector32.y);
    }

    public static Vector3 forward() {
        Vector3 vector3 = new Vector3();
        vector3.setForward();
        return vector3;
    }

    public static Vector3 left() {
        Vector3 vector3 = new Vector3();
        vector3.setLeft();
        return vector3;
    }

    public static Vector3 lerp(Vector3 vector3, Vector3 vector32, float f2) {
        Preconditions.checkNotNull(vector3, "Parameter \"a\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"b\" was null.");
        return new Vector3(MathHelper.lerp(vector3.x, vector32.x, f2), MathHelper.lerp(vector3.y, vector32.y, f2), MathHelper.lerp(vector3.z, vector32.z, f2));
    }

    public static Vector3 max(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"rhs\" was null.");
        return new Vector3(Math.max(vector3.x, vector32.x), Math.max(vector3.y, vector32.y), Math.max(vector3.z, vector32.z));
    }

    public static Vector3 min(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"rhs\" was null.");
        return new Vector3(Math.min(vector3.x, vector32.x), Math.min(vector3.y, vector32.y), Math.min(vector3.z, vector32.z));
    }

    public static Vector3 one() {
        Vector3 vector3 = new Vector3();
        vector3.setOne();
        return vector3;
    }

    public static Vector3 right() {
        Vector3 vector3 = new Vector3();
        vector3.setRight();
        return vector3;
    }

    public static Vector3 subtract(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"lhs\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"rhs\" was null.");
        return new Vector3(vector3.x - vector32.x, vector3.y - vector32.y, vector3.z - vector32.z);
    }

    public static Vector3 up() {
        Vector3 vector3 = new Vector3();
        vector3.setUp();
        return vector3;
    }

    public static Vector3 zero() {
        return new Vector3();
    }

    public int hashCode() {
        int floatToIntBits = Float.floatToIntBits(this.y);
        return Float.floatToIntBits(this.z) + ((floatToIntBits + ((Float.floatToIntBits(this.x) + 31) * 31)) * 31);
    }

    public float length() {
        return (float) Math.sqrt(lengthSquared());
    }

    public float lengthSquared() {
        float f2 = this.x;
        float f3 = this.y;
        float f4 = (f3 * f3) + (f2 * f2);
        float f5 = this.z;
        return (f5 * f5) + f4;
    }

    public Vector3 negated() {
        return new Vector3(-this.x, -this.y, -this.z);
    }

    public Vector3 normalized() {
        Vector3 vector3 = new Vector3(this);
        float dot = dot(this, this);
        if (MathHelper.almostEqualRelativeAndAbs(dot, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            vector3.setZero();
        } else if (dot != 1.0f) {
            vector3.set(scaled((float) (1.0d / Math.sqrt(dot))));
        }
        return vector3;
    }

    public Vector3 scaled(float f2) {
        return new Vector3(this.x * f2, this.y * f2, this.z * f2);
    }

    public void set(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"v\" was null.");
        this.x = vector3.x;
        this.y = vector3.y;
        this.z = vector3.z;
    }

    public void setBack() {
        set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
    }

    public void setDown() {
        set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public void setForward() {
        set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -1.0f);
    }

    public void setLeft() {
        set(-1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public void setOne() {
        set(1.0f, 1.0f, 1.0f);
    }

    public void setRight() {
        set(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public void setUp() {
        set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public void setZero() {
        set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public String toString() {
        StringBuilder x = a.x("[x=");
        x.append(this.x);
        x.append(", y=");
        x.append(this.y);
        x.append(", z=");
        x.append(this.z);
        x.append("]");
        return x.toString();
    }

    public Vector3(float f2, float f3, float f4) {
        this.x = f2;
        this.y = f3;
        this.z = f4;
    }

    public void set(float f2, float f3, float f4) {
        this.x = f2;
        this.y = f3;
        this.z = f4;
    }

    public boolean equals(Object obj) {
        if (obj instanceof Vector3) {
            if (this == obj) {
                return true;
            }
            return equals(this, (Vector3) obj);
        }
        return false;
    }

    public Vector3(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"v\" was null.");
        set(vector3);
    }
}