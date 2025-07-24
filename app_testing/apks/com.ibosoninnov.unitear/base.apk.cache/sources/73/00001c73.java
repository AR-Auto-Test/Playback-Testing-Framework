package com.google.ar.sceneform.collision;

import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class Box extends CollisionShape {
    private static final String TAG = "Box";
    private final Vector3 center;
    private final Matrix rotationMatrix;
    private final Vector3 size;

    public Box() {
        this.center = Vector3.zero();
        this.size = Vector3.one();
        this.rotationMatrix = new Matrix();
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public boolean boxIntersection(Box box) {
        return Intersections.boxBoxIntersection(this, box);
    }

    public Vector3 getCenter() {
        return new Vector3(this.center);
    }

    public Vector3 getExtents() {
        return getSize().scaled(0.5f);
    }

    public Matrix getRawRotationMatrix() {
        return this.rotationMatrix;
    }

    public Quaternion getRotation() {
        Quaternion quaternion = new Quaternion();
        this.rotationMatrix.extractQuaternion(quaternion);
        return quaternion;
    }

    public Vector3 getSize() {
        return new Vector3(this.size);
    }

    /* JADX WARN: Code restructure failed: missing block: B:14:0x0077, code lost:
        if ((r8 + r5.x) >= com.google.android.material.internal.StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) goto L10;
     */
    /* JADX WARN: Code restructure failed: missing block: B:28:0x00c3, code lost:
        if ((r8 + r5.y) >= com.google.android.material.internal.StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) goto L18;
     */
    /* JADX WARN: Code restructure failed: missing block: B:42:0x0111, code lost:
        if ((r3 + r5.z) >= com.google.android.material.internal.StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) goto L26;
     */
    @Override // com.google.ar.sceneform.collision.CollisionShape
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean rayIntersection(Ray ray, RayHit rayHit) {
        Preconditions.checkNotNull(ray, "Parameter \"ray\" was null.");
        Preconditions.checkNotNull(rayHit, "Parameter \"result\" was null.");
        Vector3 direction = ray.getDirection();
        Vector3 origin = ray.getOrigin();
        Vector3 extents = getExtents();
        Vector3 negated = extents.negated();
        Vector3 subtract = Vector3.subtract(this.center, origin);
        float[] fArr = this.rotationMatrix.data;
        Vector3 vector3 = new Vector3(fArr[0], fArr[1], fArr[2]);
        float dot = Vector3.dot(vector3, subtract);
        float dot2 = Vector3.dot(direction, vector3);
        float f2 = Float.MIN_VALUE;
        float f3 = Float.MAX_VALUE;
        if (!MathHelper.almostEqualRelativeAndAbs(dot2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            float f4 = (negated.x + dot) / dot2;
            float f5 = (dot + extents.x) / dot2;
            if (f4 <= f5) {
                f4 = f5;
                f5 = f4;
            }
            f3 = Math.min(f4, Float.MAX_VALUE);
            f2 = Math.max(f5, Float.MIN_VALUE);
            if (f3 < f2) {
                return false;
            }
        } else {
            float f6 = -dot;
            if (negated.x + f6 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            }
            return false;
        }
        Vector3 vector32 = new Vector3(fArr[4], fArr[5], fArr[6]);
        float dot3 = Vector3.dot(vector32, subtract);
        float dot4 = Vector3.dot(direction, vector32);
        if (!MathHelper.almostEqualRelativeAndAbs(dot4, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            float f7 = (negated.y + dot3) / dot4;
            float f8 = (dot3 + extents.y) / dot4;
            if (f7 <= f8) {
                f7 = f8;
                f8 = f7;
            }
            f3 = Math.min(f7, f3);
            f2 = Math.max(f8, f2);
            if (f3 < f2) {
                return false;
            }
        } else {
            float f9 = -dot3;
            if (negated.y + f9 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            }
            return false;
        }
        Vector3 vector33 = new Vector3(fArr[8], fArr[9], fArr[10]);
        float dot5 = Vector3.dot(vector33, subtract);
        float dot6 = Vector3.dot(direction, vector33);
        if (!MathHelper.almostEqualRelativeAndAbs(dot6, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            float f10 = (negated.z + dot5) / dot6;
            float f11 = (dot5 + extents.z) / dot6;
            if (f10 <= f11) {
                f10 = f11;
                f11 = f10;
            }
            float min = Math.min(f10, f3);
            f2 = Math.max(f11, f2);
            if (min < f2) {
                return false;
            }
        } else {
            float f12 = -dot5;
            if (negated.z + f12 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            }
            return false;
        }
        rayHit.setDistance(f2);
        rayHit.setPoint(ray.getPoint(rayHit.getDistance()));
        return true;
    }

    public void setCenter(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"center\" was null.");
        this.center.set(vector3);
        onChanged();
    }

    public void setRotation(Quaternion quaternion) {
        Preconditions.checkNotNull(quaternion, "Parameter \"rotation\" was null.");
        this.rotationMatrix.makeRotation(quaternion);
        onChanged();
    }

    public void setSize(Vector3 vector3) {
        Preconditions.checkNotNull(vector3, "Parameter \"size\" was null.");
        this.size.set(vector3);
        onChanged();
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public boolean shapeIntersection(CollisionShape collisionShape) {
        Preconditions.checkNotNull(collisionShape, "Parameter \"shape\" was null.");
        return collisionShape.boxIntersection(this);
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public boolean sphereIntersection(Sphere sphere) {
        return Intersections.sphereBoxIntersection(sphere, this);
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public CollisionShape transform(TransformProvider transformProvider) {
        Preconditions.checkNotNull(transformProvider, "Parameter \"transformProvider\" was null.");
        Box box = new Box();
        transform(transformProvider, box);
        return box;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.ar.sceneform.collision.CollisionShape
    public Box makeCopy() {
        return new Box(getSize(), getCenter());
    }

    @Override // com.google.ar.sceneform.collision.CollisionShape
    public void transform(TransformProvider transformProvider, CollisionShape collisionShape) {
        Preconditions.checkNotNull(transformProvider, "Parameter \"transformProvider\" was null.");
        Preconditions.checkNotNull(collisionShape, "Parameter \"result\" was null.");
        if (!(collisionShape instanceof Box)) {
            Log.w(TAG, "Cannot pass CollisionShape of a type other than Box into Box.transform.");
        } else if (collisionShape != this) {
            Box box = (Box) collisionShape;
            Matrix worldModelMatrix = transformProvider.getWorldModelMatrix();
            box.center.set(worldModelMatrix.transformPoint(this.center));
            Vector3 vector3 = new Vector3();
            worldModelMatrix.decomposeScale(vector3);
            Vector3 vector32 = box.size;
            Vector3 vector33 = this.size;
            vector32.x = vector33.x * vector3.x;
            vector32.y = vector33.y * vector3.y;
            vector32.z = vector33.z * vector3.z;
            worldModelMatrix.decomposeRotation(vector3, box.rotationMatrix);
            Matrix matrix = this.rotationMatrix;
            Matrix matrix2 = box.rotationMatrix;
            Matrix.multiply(matrix, matrix2, matrix2);
        } else {
            throw new IllegalArgumentException("Box cannot transform itself.");
        }
    }

    public Box(Vector3 vector3) {
        this(vector3, Vector3.zero());
    }

    public Box(Vector3 vector3, Vector3 vector32) {
        this.center = Vector3.zero();
        this.size = Vector3.one();
        this.rotationMatrix = new Matrix();
        Preconditions.checkNotNull(vector32, "Parameter \"center\" was null.");
        Preconditions.checkNotNull(vector3, "Parameter \"size\" was null.");
        setCenter(vector32);
        setSize(vector3);
    }
}