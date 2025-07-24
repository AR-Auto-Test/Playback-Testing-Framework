package com.google.ar.sceneform.collision;

import c.b.a.a.a;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Ray {
    private Vector3 origin = new Vector3();
    private Vector3 direction = Vector3.forward();

    public Ray() {
    }

    public Vector3 getDirection() {
        return new Vector3(this.direction);
    }

    public Vector3 getOrigin() {
        return new Vector3(this.origin);
    }

    public Vector3 getPoint(float f2) {
        return Vector3.add(this.origin, this.direction.scaled(f2));
    }

    public void setDirection(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"direction\" was null.");
        this.direction.set(vector3.normalized());
    }

    public void setOrigin(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"origin\" was null.");
        this.origin.set(vector3);
    }

    public String toString() {
        StringBuilder x = a.x("[Origin:");
        x.append(this.origin);
        x.append(", Direction:");
        x.append(this.direction);
        x.append("]");
        return x.toString();
    }

    public Ray(Vector3 vector3, Vector3 vector32) {
        Preconditions.checkNotNull(vector3, "Parameter \"origin\" was null.");
        Preconditions.checkNotNull(vector32, "Parameter \"direction\" was null.");
        setOrigin(vector3);
        setDirection(vector32);
    }
}