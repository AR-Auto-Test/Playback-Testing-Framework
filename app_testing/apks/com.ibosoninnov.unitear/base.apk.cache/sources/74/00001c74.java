package com.google.ar.sceneform.collision;

import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Collider {
    private CollisionSystem attachedCollisionSystem;
    private CollisionShape cachedWorldShape;
    private boolean isWorldShapeDirty;
    private CollisionShape localShape;
    private int shapeId = 0;
    private TransformProvider transformProvider;

    public Collider(TransformProvider transformProvider, CollisionShape collisionShape) {
        Preconditions.checkNotNull(transformProvider, "Parameter \"transformProvider\" was null.");
        Preconditions.checkNotNull(collisionShape, "Parameter \"localCollisionShape\" was null.");
        this.transformProvider = transformProvider;
        setShape(collisionShape);
    }

    private boolean doesCachedWorldShapeNeedUpdate() {
        CollisionShape collisionShape = this.localShape;
        if (collisionShape == null) {
            return false;
        }
        return collisionShape.getId().checkChanged(this.shapeId) || this.isWorldShapeDirty || this.cachedWorldShape == null;
    }

    private void updateCachedWorldShape() {
        if (doesCachedWorldShapeNeedUpdate()) {
            CollisionShape collisionShape = this.cachedWorldShape;
            if (collisionShape == null) {
                this.cachedWorldShape = this.localShape.transform(this.transformProvider);
            } else {
                this.localShape.transform(this.transformProvider, collisionShape);
            }
            this.shapeId = this.localShape.getId().get();
        }
    }

    public CollisionShape getShape() {
        return this.localShape;
    }

    public TransformProvider getTransformProvider() {
        return this.transformProvider;
    }

    public CollisionShape getTransformedShape() {
        updateCachedWorldShape();
        return this.cachedWorldShape;
    }

    public void markWorldShapeDirty() {
        this.isWorldShapeDirty = true;
    }

    public void setAttachedCollisionSystem(CollisionSystem collisionSystem) {
        CollisionSystem collisionSystem2 = this.attachedCollisionSystem;
        if (collisionSystem2 != null) {
            collisionSystem2.removeCollider(this);
        }
        this.attachedCollisionSystem = collisionSystem;
        if (collisionSystem != null) {
            collisionSystem.addCollider(this);
        }
    }

    public void setShape(CollisionShape collisionShape) {
        Preconditions.checkNotNull(collisionShape, "Parameter \"localCollisionShape\" was null.");
        this.localShape = collisionShape;
        this.cachedWorldShape = null;
    }
}