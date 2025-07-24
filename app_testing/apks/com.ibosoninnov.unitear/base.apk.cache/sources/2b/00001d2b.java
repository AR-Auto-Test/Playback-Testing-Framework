package com.google.ar.sceneform.rendering;

import com.google.ar.sceneform.math.Vector3;

/* loaded from: classes.dex */
public class Vertex {
    private Color color;
    private Vector3 normal;
    private final Vector3 position;
    private UvCoordinate uvCoordinate;

    /* loaded from: classes.dex */
    public static final class Builder {
        private Color color;
        private Vector3 normal;
        private final Vector3 position = Vector3.zero();
        private UvCoordinate uvCoordinate;

        public Vertex build() {
            return new Vertex(this);
        }

        public Builder setColor(Color color) {
            this.color = color;
            return this;
        }

        public Builder setNormal(Vector3 vector3) {
            this.normal = vector3;
            return this;
        }

        public Builder setPosition(Vector3 vector3) {
            this.position.set(vector3);
            return this;
        }

        public Builder setUvCoordinate(UvCoordinate uvCoordinate) {
            this.uvCoordinate = uvCoordinate;
            return this;
        }
    }

    /* loaded from: classes.dex */
    public static final class UvCoordinate {
        public float x;
        public float y;

        public UvCoordinate(float f2, float f3) {
            this.x = f2;
            this.y = f3;
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public Color getColor() {
        return this.color;
    }

    public Vector3 getNormal() {
        return this.normal;
    }

    public Vector3 getPosition() {
        return this.position;
    }

    public UvCoordinate getUvCoordinate() {
        return this.uvCoordinate;
    }

    public void setColor(Color color) {
        this.color = color;
    }

    public void setNormal(Vector3 vector3) {
        this.normal = vector3;
    }

    public void setPosition(Vector3 vector3) {
        this.position.set(vector3);
    }

    public void setUvCoordinate(UvCoordinate uvCoordinate) {
        this.uvCoordinate = uvCoordinate;
    }

    private Vertex(Builder builder) {
        Vector3 zero = Vector3.zero();
        this.position = zero;
        zero.set(builder.position);
        this.normal = builder.normal;
        this.uvCoordinate = builder.uvCoordinate;
        this.color = builder.color;
    }
}