package com.google.ar.sceneform.collision;

import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Sphere extends CollisionShape {
    private static final String TAG = "Sphere";
    private final Vector3 center;
    private float radius;

    public Sphere() {
        this.center = new Vector3();
        this.radius = 1.0f;
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public boolean boxIntersection(Box box) {
        return Intersections.sphereBoxIntersection(this, box);
    }

    public Vector3 getCenter() {
        return new Vector3(this.center);
    }

    public float getRadius() {
        return this.radius;
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public boolean rayIntersection(Ray ray, RayHit rayHit) {
        Preconditions.checkNotNull(ray, "Parameter \"ray\" was null.");
        Preconditions.checkNotNull(rayHit, "Parameter \"result\" was null.");
        Vector3 direction = ray.getDirection();
        Vector3 subtract = Vector3.subtract(ray.getOrigin(), this.center);
        float dot = Vector3.dot(subtract, direction) * 2.0f;
        float dot2 = Vector3.dot(subtract, subtract);
        float f2 = this.radius;
        float f3 = (dot * dot) - ((dot2 - (f2 * f2)) * 4.0f);
        if (f3 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return false;
        }
        float sqrt = (float) Math.sqrt(f3);
        float f4 = -dot;
        float f5 = (f4 - sqrt) / 2.0f;
        float f6 = (f4 + sqrt) / 2.0f;
        int i = (f5 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (f5 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
        if (i >= 0 || f6 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            if (i < 0 && f6 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                rayHit.setDistance(f6);
            } else {
                rayHit.setDistance(f5);
            }
            rayHit.setPoint(ray.getPoint(rayHit.getDistance()));
            return true;
        }
        return false;
    }

    public void setCenter(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"center\" was null.");
        this.center.set(vector3);
        onChanged();
    }

    public void setRadius(float f2) {
        this.radius = f2;
        onChanged();
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public boolean shapeIntersection(CollisionShape collisionShape) {
        Preconditions.checkNotNull(collisionShape, "Parameter \"shape\" was null.");
        return collisionShape.sphereIntersection(this);
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public boolean sphereIntersection(Sphere sphere) {
        return Intersections.sphereSphereIntersection(this, sphere);
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public CollisionShape transform(TransformProvider transformProvider) {
        Preconditions.checkNotNull(transformProvider, "Parameter \"transformProvider\" was null.");
        Sphere sphere = new Sphere();
        transform(transformProvider, sphere);
        return sphere;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.collision.CollisionShape
    public Sphere makeCopy() {
        return new Sphere(getRadius(), getCenter());
    }

    public Sphere(float f2) {
        this(f2, Vector3.zero());
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public void transform(TransformProvider transformProvider, CollisionShape collisionShape) {
        Preconditions.checkNotNull(transformProvider, "Parameter \"transformProvider\" was null.");
        Preconditions.checkNotNull(collisionShape, "Parameter \"result\" was null.");
        if (!(collisionShape instanceof Sphere)) {
            Log.w(TAG, "Cannot pass CollisionShape of a type other than Sphere into Sphere.transform.");
            return;
        }
        Sphere sphere = (Sphere) collisionShape;
        Matrix worldModelMatrix = transformProvider.getWorldModelMatrix();
        sphere.setCenter(worldModelMatrix.transformPoint(this.center));
        Vector3 vector3 = new Vector3();
        worldModelMatrix.decomposeScale(vector3);
        sphere.radius = this.radius * Math.max(Math.abs(Math.min(Math.min(vector3.x, vector3.y), vector3.z)), Math.max(Math.max(vector3.x, vector3.y), vector3.z));
    }

    public Sphere(float f2, Vector3 vector3) {
        this.center = new Vector3();
        this.radius = 1.0f;
        Preconditions.checkNotNull(vector3, "Parameter \"center\" was null.");
        setCenter(vector3);
        setRadius(f2);
    }
}