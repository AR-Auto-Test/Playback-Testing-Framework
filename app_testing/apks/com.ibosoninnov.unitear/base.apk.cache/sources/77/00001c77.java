package com.google.ar.sceneform.collision;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes.dex */
public class Intersections {
    private static final int NUM_TEST_AXES = 15;
    private static final int NUM_VERTICES_PER_BOX = 8;

    public static boolean boxBoxIntersection(Box box, Box box2) {
        Preconditions.checkNotNull(box, "Parameter \"box1\" was null.");
        Preconditions.checkNotNull(box2, "Parameter \"box2\" was null.");
        List<Vector3> verticesFromBox = getVerticesFromBox(box);
        List<Vector3> verticesFromBox2 = getVerticesFromBox(box2);
        Matrix rawRotationMatrix = box.getRawRotationMatrix();
        Matrix rawRotationMatrix2 = box2.getRawRotationMatrix();
        ArrayList arrayList = new ArrayList(15);
        arrayList.add(extractXAxisFromRotationMatrix(rawRotationMatrix));
        arrayList.add(extractYAxisFromRotationMatrix(rawRotationMatrix));
        arrayList.add(extractZAxisFromRotationMatrix(rawRotationMatrix));
        arrayList.add(extractXAxisFromRotationMatrix(rawRotationMatrix2));
        arrayList.add(extractYAxisFromRotationMatrix(rawRotationMatrix2));
        arrayList.add(extractZAxisFromRotationMatrix(rawRotationMatrix2));
        for (int i = 0; i < 3; i++) {
            arrayList.add(Vector3.cross((Vector3) arrayList.get(i), (Vector3) arrayList.get(0)));
            arrayList.add(Vector3.cross((Vector3) arrayList.get(i), (Vector3) arrayList.get(1)));
            arrayList.add(Vector3.cross((Vector3) arrayList.get(i), (Vector3) arrayList.get(2)));
        }
        for (int i2 = 0; i2 < arrayList.size(); i2++) {
            if (!testSeparatingAxis(verticesFromBox, verticesFromBox2, (Vector3) arrayList.get(i2))) {
                return false;
            }
        }
        return true;
    }

    private static Vector3 closestPointOnBox(Vector3 vector3, Box box) {
        Vector3 vector32 = new Vector3(box.getCenter());
        Vector3 subtract = Vector3.subtract(vector3, box.getCenter());
        Matrix rawRotationMatrix = box.getRawRotationMatrix();
        Vector3 extents = box.getExtents();
        Vector3 extractXAxisFromRotationMatrix = extractXAxisFromRotationMatrix(rawRotationMatrix);
        float dot = Vector3.dot(subtract, extractXAxisFromRotationMatrix);
        float f2 = extents.x;
        if (dot > f2) {
            dot = f2;
        } else if (dot < (-f2)) {
            dot = -f2;
        }
        Vector3 add = Vector3.add(vector32, extractXAxisFromRotationMatrix.scaled(dot));
        Vector3 extractYAxisFromRotationMatrix = extractYAxisFromRotationMatrix(rawRotationMatrix);
        float dot2 = Vector3.dot(subtract, extractYAxisFromRotationMatrix);
        float f3 = extents.y;
        if (dot2 > f3) {
            dot2 = f3;
        } else if (dot2 < (-f3)) {
            dot2 = -f3;
        }
        Vector3 add2 = Vector3.add(add, extractYAxisFromRotationMatrix.scaled(dot2));
        Vector3 extractZAxisFromRotationMatrix = extractZAxisFromRotationMatrix(rawRotationMatrix);
        float dot3 = Vector3.dot(subtract, extractZAxisFromRotationMatrix);
        float f4 = extents.z;
        if (dot3 > f4) {
            dot3 = f4;
        } else if (dot3 < (-f4)) {
            dot3 = -f4;
        }
        return Vector3.add(add2, extractZAxisFromRotationMatrix.scaled(dot3));
    }

    private static Vector3 extractXAxisFromRotationMatrix(Matrix matrix) {
        float[] fArr = matrix.data;
        return new Vector3(fArr[0], fArr[4], fArr[8]);
    }

    private static Vector3 extractYAxisFromRotationMatrix(Matrix matrix) {
        float[] fArr = matrix.data;
        return new Vector3(fArr[1], fArr[5], fArr[9]);
    }

    private static Vector3 extractZAxisFromRotationMatrix(Matrix matrix) {
        float[] fArr = matrix.data;
        return new Vector3(fArr[2], fArr[6], fArr[10]);
    }

    private static List<Vector3> getVerticesFromBox(Box box) {
        Preconditions.checkNotNull(box, "Parameter \"box\" was null.");
        Vector3 center = box.getCenter();
        Vector3 extents = box.getExtents();
        Matrix rawRotationMatrix = box.getRawRotationMatrix();
        Vector3 extractXAxisFromRotationMatrix = extractXAxisFromRotationMatrix(rawRotationMatrix);
        Vector3 extractYAxisFromRotationMatrix = extractYAxisFromRotationMatrix(rawRotationMatrix);
        Vector3 extractZAxisFromRotationMatrix = extractZAxisFromRotationMatrix(rawRotationMatrix);
        Vector3 scaled = extractXAxisFromRotationMatrix.scaled(extents.x);
        Vector3 scaled2 = extractYAxisFromRotationMatrix.scaled(extents.y);
        Vector3 scaled3 = extractZAxisFromRotationMatrix.scaled(extents.z);
        ArrayList arrayList = new ArrayList(8);
        arrayList.add(Vector3.add(Vector3.add(Vector3.add(center, scaled), scaled2), scaled3));
        arrayList.add(Vector3.add(Vector3.add(Vector3.subtract(center, scaled), scaled2), scaled3));
        arrayList.add(Vector3.add(Vector3.subtract(Vector3.add(center, scaled), scaled2), scaled3));
        arrayList.add(Vector3.subtract(Vector3.add(Vector3.add(center, scaled), scaled2), scaled3));
        arrayList.add(Vector3.subtract(Vector3.subtract(Vector3.subtract(center, scaled), scaled2), scaled3));
        arrayList.add(Vector3.subtract(Vector3.subtract(Vector3.add(center, scaled), scaled2), scaled3));
        arrayList.add(Vector3.subtract(Vector3.add(Vector3.subtract(center, scaled), scaled2), scaled3));
        arrayList.add(Vector3.add(Vector3.subtract(Vector3.subtract(center, scaled), scaled2), scaled3));
        return arrayList;
    }

    public static boolean sphereBoxIntersection(Sphere sphere, Box box) {
        Preconditions.checkNotNull(sphere, "Parameter \"sphere\" was null.");
        Preconditions.checkNotNull(box, "Parameter \"box\" was null.");
        Vector3 closestPointOnBox = closestPointOnBox(sphere.getCenter(), box);
        Vector3 subtract = Vector3.subtract(closestPointOnBox, sphere.getCenter());
        float dot = Vector3.dot(subtract, subtract);
        if (dot > sphere.getRadius() * sphere.getRadius()) {
            return false;
        }
        if (MathHelper.almostEqualRelativeAndAbs(dot, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            Vector3 subtract2 = Vector3.subtract(closestPointOnBox, box.getCenter());
            return !MathHelper.almostEqualRelativeAndAbs(Vector3.dot(subtract2, subtract2), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        }
        return true;
    }

    public static boolean sphereSphereIntersection(Sphere sphere, Sphere sphere2) {
        Preconditions.checkNotNull(sphere, "Parameter \"sphere1\" was null.");
        Preconditions.checkNotNull(sphere2, "Parameter \"sphere2\" was null.");
        float radius = sphere2.getRadius() + sphere.getRadius();
        Vector3 subtract = Vector3.subtract(sphere2.getCenter(), sphere.getCenter());
        float dot = Vector3.dot(subtract, subtract);
        return dot - (radius * radius) <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && dot != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    private static boolean testSeparatingAxis(List<Vector3> list, List<Vector3> list2, Vector3 vector3) {
        float f2 = Float.MIN_VALUE;
        float f3 = Float.MAX_VALUE;
        float f4 = Float.MIN_VALUE;
        float f5 = Float.MAX_VALUE;
        for (int i = 0; i < list.size(); i++) {
            float dot = Vector3.dot(vector3, list.get(i));
            f5 = Math.min(dot, f5);
            f4 = Math.max(dot, f4);
        }
        for (int i2 = 0; i2 < list2.size(); i2++) {
            float dot2 = Vector3.dot(vector3, list2.get(i2));
            f3 = Math.min(dot2, f3);
            f2 = Math.max(dot2, f2);
        }
        return f3 <= f4 && f5 <= f2;
    }
}