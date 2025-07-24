package com.google.ar.sceneform.collision;

import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.utilities.ChangeId;

/* loaded from: classes.dex */
public abstract class CollisionShape {
    private final ChangeId changeId;

    public CollisionShape() {
        ChangeId changeId = new ChangeId();
        this.changeId = changeId;
        changeId.update();
    }

    public abstract boolean boxIntersection(Box box);

    public ChangeId getId() {
        return this.changeId;
    }

    public abstract CollisionShape makeCopy();

    public void onChanged() {
        this.changeId.update();
    }

    public abstract boolean rayIntersection(Ray ray, RayHit rayHit);

    public abstract boolean shapeIntersection(CollisionShape collisionShape);

    public abstract boolean sphereIntersection(Sphere sphere);

    public abstract CollisionShape transform(TransformProvider transformProvider);

    public abstract void transform(TransformProvider transformProvider, CollisionShape collisionShape);
}