package com.google.ar.sceneform.collision;

import c.d.b.a.p.a;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;

/* loaded from: classes.dex */
public class CollisionSystem {
    private static final String TAG = "CollisionSystem";
    private final ArrayList<Collider> colliders = new ArrayList<>();

    public void addCollider(Collider collider) {
        Preconditions.checkNotNull(collider, "Parameter \"collider\" was null.");
        this.colliders.add(collider);
    }

    public Collider intersects(Collider collider) {
        CollisionShape transformedShape;
        Preconditions.checkNotNull(collider, "Parameter \"collider\" was null.");
        CollisionShape transformedShape2 = collider.getTransformedShape();
        if (transformedShape2 == null) {
            return null;
        }
        Iterator<Collider> it = this.colliders.iterator();
        while (it.hasNext()) {
            Collider next = it.next();
            if (next != collider && (transformedShape = next.getTransformedShape()) != null && transformedShape2.shapeIntersection(transformedShape)) {
                return next;
            }
        }
        return null;
    }

    public void intersectsAll(Collider collider, Consumer<Collider> consumer) {
        CollisionShape transformedShape;
        Preconditions.checkNotNull(collider, "Parameter \"collider\" was null.");
        Preconditions.checkNotNull(consumer, "Parameter \"processResult\" was null.");
        CollisionShape transformedShape2 = collider.getTransformedShape();
        if (transformedShape2 == null) {
            return;
        }
        Iterator<Collider> it = this.colliders.iterator();
        while (it.hasNext()) {
            Collider next = it.next();
            if (next != collider && (transformedShape = next.getTransformedShape()) != null && transformedShape2.shapeIntersection(transformedShape)) {
                consumer.accept(next);
            }
        }
    }

    public Collider raycast(Ray ray, RayHit rayHit) {
        Preconditions.checkNotNull(ray, "Parameter \"ray\" was null.");
        Preconditions.checkNotNull(rayHit, "Parameter \"resultHit\" was null.");
        rayHit.reset();
        RayHit rayHit2 = new RayHit();
        Iterator<Collider> it = this.colliders.iterator();
        Collider collider = null;
        while (it.hasNext()) {
            Collider next = it.next();
            CollisionShape transformedShape = next.getTransformedShape();
            if (transformedShape != null && transformedShape.rayIntersection(ray, rayHit2) && rayHit2.getDistance() < rayHit.getDistance()) {
                rayHit.set(rayHit2);
                collider = next;
            }
        }
        return collider;
    }

    public <T extends RayHit> int raycastAll(Ray ray, ArrayList<T> arrayList, BiConsumer<T, Collider> biConsumer, Supplier<T> supplier) {
        T t;
        Preconditions.checkNotNull(ray, "Parameter \"ray\" was null.");
        Preconditions.checkNotNull(arrayList, "Parameter \"resultBuffer\" was null.");
        Preconditions.checkNotNull(supplier, "Parameter \"allocateResult\" was null.");
        RayHit rayHit = new RayHit();
        Iterator<Collider> it = this.colliders.iterator();
        int i = 0;
        while (it.hasNext()) {
            Collider next = it.next();
            CollisionShape transformedShape = next.getTransformedShape();
            if (transformedShape != null && transformedShape.rayIntersection(ray, rayHit)) {
                i++;
                if (arrayList.size() >= i) {
                    t = arrayList.get(i - 1);
                } else {
                    t = supplier.get();
                    arrayList.add(t);
                }
                t.reset();
                t.set(rayHit);
                if (biConsumer != null) {
                    biConsumer.accept(t, next);
                }
            }
        }
        for (int i2 = i; i2 < arrayList.size(); i2++) {
            arrayList.get(i2).reset();
        }
        Collections.sort(arrayList, a.f4312b);
        return i;
    }

    public void removeCollider(Collider collider) {
        Preconditions.checkNotNull(collider, "Parameter \"collider\" was null.");
        this.colliders.remove(collider);
    }
}