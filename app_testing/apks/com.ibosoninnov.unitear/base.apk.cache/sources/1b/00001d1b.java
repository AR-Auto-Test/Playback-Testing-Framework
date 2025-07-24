package com.google.ar.sceneform.rendering;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.RenderableDefinition;
import com.google.ar.sceneform.rendering.Vertex;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;

/* loaded from: classes.dex */
public final class ShapeFactory {
    private static final int COORDS_PER_TRIANGLE = 3;
    private static final String TAG = "ShapeFactory";

    public static ModelRenderable makeCube(Vector3 vector3, Vector3 vector32, Material material) {
        AndroidPreconditions.checkMinAndroidApiLevel();
        Vector3 scaled = vector3.scaled(0.5f);
        Vector3 add = Vector3.add(vector32, new Vector3(-scaled.x, -scaled.y, scaled.z));
        Vector3 add2 = Vector3.add(vector32, new Vector3(scaled.x, -scaled.y, scaled.z));
        Vector3 add3 = Vector3.add(vector32, new Vector3(scaled.x, -scaled.y, -scaled.z));
        Vector3 add4 = Vector3.add(vector32, new Vector3(-scaled.x, -scaled.y, -scaled.z));
        Vector3 add5 = Vector3.add(vector32, new Vector3(-scaled.x, scaled.y, scaled.z));
        Vector3 add6 = Vector3.add(vector32, new Vector3(scaled.x, scaled.y, scaled.z));
        Vector3 add7 = Vector3.add(vector32, new Vector3(scaled.x, scaled.y, -scaled.z));
        Vector3 add8 = Vector3.add(vector32, new Vector3(-scaled.x, scaled.y, -scaled.z));
        Vector3 up = Vector3.up();
        Vector3 down = Vector3.down();
        Vector3 forward = Vector3.forward();
        Vector3 back = Vector3.back();
        Vector3 left = Vector3.left();
        Vector3 right = Vector3.right();
        Vertex.UvCoordinate uvCoordinate = new Vertex.UvCoordinate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        Vertex.UvCoordinate uvCoordinate2 = new Vertex.UvCoordinate(1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        Vertex.UvCoordinate uvCoordinate3 = new Vertex.UvCoordinate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
        Vertex.UvCoordinate uvCoordinate4 = new Vertex.UvCoordinate(1.0f, 1.0f);
        ArrayList arrayList = new ArrayList(Arrays.asList(a.R(add, down, uvCoordinate3), a.R(add2, down, uvCoordinate4), a.R(add3, down, uvCoordinate2), a.R(add4, down, uvCoordinate), a.R(add8, left, uvCoordinate3), a.R(add5, left, uvCoordinate4), a.R(add, left, uvCoordinate2), a.R(add4, left, uvCoordinate), a.R(add5, forward, uvCoordinate3), a.R(add6, forward, uvCoordinate4), a.R(add2, forward, uvCoordinate2), a.R(add, forward, uvCoordinate), a.R(add7, back, uvCoordinate3), a.R(add8, back, uvCoordinate4), a.R(add4, back, uvCoordinate2), a.R(add3, back, uvCoordinate), a.R(add6, right, uvCoordinate3), a.R(add7, right, uvCoordinate4), a.R(add3, right, uvCoordinate2), a.R(add2, right, uvCoordinate), a.R(add8, up, uvCoordinate3), a.R(add7, up, uvCoordinate4), a.R(add6, up, uvCoordinate2), a.R(add5, up, uvCoordinate)));
        ArrayList arrayList2 = new ArrayList(36);
        for (int i = 0; i < 6; i++) {
            int i2 = i * 4;
            int i3 = i2 + 3;
            arrayList2.add(Integer.valueOf(i3));
            int i4 = i2 + 1;
            arrayList2.add(Integer.valueOf(i4));
            arrayList2.add(Integer.valueOf(i2 + 0));
            arrayList2.add(Integer.valueOf(i3));
            arrayList2.add(Integer.valueOf(i2 + 2));
            arrayList2.add(Integer.valueOf(i4));
        }
        try {
            ModelRenderable modelRenderable = ModelRenderable.builder().setSource(RenderableDefinition.builder().setVertices(arrayList).setSubmeshes(Arrays.asList(RenderableDefinition.Submesh.builder().setTriangleIndices(arrayList2).setMaterial(material).build())).build()).build().get();
            if (modelRenderable != null) {
                return modelRenderable;
            }
            throw new AssertionError("Error creating renderable.");
        } catch (InterruptedException | ExecutionException e2) {
            throw new AssertionError("Error creating renderable.", e2);
        }
    }

    public static ModelRenderable makeCylinder(float f2, float f3, Vector3 vector3, Material material) {
        String str = "Error creating renderable.";
        AndroidPreconditions.checkMinAndroidApiLevel();
        float f4 = f3 / 2.0f;
        ArrayList arrayList = new ArrayList(100);
        ArrayList arrayList2 = new ArrayList(25);
        ArrayList arrayList3 = new ArrayList(25);
        ArrayList arrayList4 = new ArrayList(25);
        float f5 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        float f6 = 0.0f;
        int i = 0;
        while (i <= 24) {
            double d2 = f6;
            float cos = (float) Math.cos(d2);
            float sin = (float) Math.sin(d2);
            float f7 = f2 * cos;
            float f8 = f2 * sin;
            Vector3 vector32 = new Vector3(f7, -f4, f8);
            String str2 = str;
            Vector3 normalized = new Vector3(vector32.x, f5, vector32.z).normalized();
            Vector3 add = Vector3.add(vector32, vector3);
            float f9 = i * 0.041666668f;
            arrayList.add(Vertex.builder().setPosition(add).setNormal(normalized).setUvCoordinate(new Vertex.UvCoordinate(f9, f5)).build());
            float f10 = (cos + 1.0f) / 2.0f;
            float f11 = (sin + 1.0f) / 2.0f;
            arrayList2.add(Vertex.builder().setPosition(add).setNormal(Vector3.down()).setUvCoordinate(new Vertex.UvCoordinate(f10, f11)).build());
            Vector3 vector33 = new Vector3(f7, f4, f8);
            Vector3 normalized2 = new Vector3(vector33.x, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, vector33.z).normalized();
            Vector3 add2 = Vector3.add(vector33, vector3);
            arrayList4.add(Vertex.builder().setPosition(add2).setNormal(normalized2).setUvCoordinate(new Vertex.UvCoordinate(f9, 1.0f)).build());
            arrayList3.add(Vertex.builder().setPosition(add2).setNormal(Vector3.up()).setUvCoordinate(new Vertex.UvCoordinate(f10, f11)).build());
            f6 += 0.2617994f;
            i++;
            str = str2;
            f5 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        String str3 = str;
        arrayList.addAll(arrayList4);
        int size = arrayList.size();
        arrayList.add(Vertex.builder().setPosition(Vector3.add(vector3, new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, -f4, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))).setNormal(Vector3.down()).setUvCoordinate(new Vertex.UvCoordinate(0.5f, 0.5f)).build());
        arrayList.addAll(arrayList2);
        int size2 = arrayList.size();
        arrayList.add(Vertex.builder().setPosition(Vector3.add(vector3, new Vector3(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f4, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))).setNormal(Vector3.up()).setUvCoordinate(new Vertex.UvCoordinate(0.5f, 0.5f)).build());
        arrayList.addAll(arrayList3);
        ArrayList arrayList5 = new ArrayList();
        int i2 = 0;
        while (i2 < 24) {
            int i3 = i2 + 1;
            int i4 = i2 + 24;
            int i5 = i4 + 1;
            int i6 = i4 + 2;
            arrayList5.add(Integer.valueOf(i2));
            arrayList5.add(Integer.valueOf(i6));
            arrayList5.add(Integer.valueOf(i3));
            arrayList5.add(Integer.valueOf(i2));
            arrayList5.add(Integer.valueOf(i5));
            arrayList5.add(Integer.valueOf(i6));
            arrayList5.add(Integer.valueOf(size));
            int i7 = size + i2;
            arrayList5.add(Integer.valueOf(i7 + 1));
            arrayList5.add(Integer.valueOf(i7 + 2));
            arrayList5.add(Integer.valueOf(size2));
            int i8 = i2 + size2;
            arrayList5.add(Integer.valueOf(i8 + 2));
            arrayList5.add(Integer.valueOf(i8 + 1));
            i2 = i3;
        }
        try {
            ModelRenderable modelRenderable = ModelRenderable.builder().setSource(RenderableDefinition.builder().setVertices(arrayList).setSubmeshes(Arrays.asList(RenderableDefinition.Submesh.builder().setTriangleIndices(arrayList5).setMaterial(material).build())).build()).build().get();
            if (modelRenderable != null) {
                return modelRenderable;
            }
            throw new AssertionError(str3);
        } catch (InterruptedException | ExecutionException e2) {
            throw new AssertionError(str3, e2);
        }
    }

    public static ModelRenderable makeSphere(float f2, Vector3 vector3, Material material) {
        AndroidPreconditions.checkMinAndroidApiLevel();
        ArrayList arrayList = new ArrayList(602);
        for (int i = 0; i <= 24; i++) {
            float f3 = i;
            float f4 = 24.0f;
            double d2 = (3.1415927f * f3) / 24.0f;
            float sin = (float) Math.sin(d2);
            float cos = (float) Math.cos(d2);
            int i2 = 0;
            while (i2 <= 24) {
                double d3 = (6.2831855f * (i2 == 24 ? 0 : i2)) / f4;
                Vector3 scaled = new Vector3(((float) Math.cos(d3)) * sin, cos, ((float) Math.sin(d3)) * sin).scaled(f2);
                arrayList.add(Vertex.builder().setPosition(Vector3.add(scaled, vector3)).setNormal(scaled.normalized()).setUvCoordinate(new Vertex.UvCoordinate(1.0f - (i2 / f4), 1.0f - (f3 / f4))).build());
                i2++;
                f4 = 24.0f;
            }
        }
        ArrayList arrayList2 = new ArrayList(arrayList.size() * 2 * 3);
        int i3 = 0;
        int i4 = 0;
        while (i3 < 24) {
            int i5 = 0;
            while (i5 < 24) {
                boolean z = i3 == 0;
                boolean z2 = i3 == 23;
                int i6 = i5 + 1;
                if (!z) {
                    int i7 = i4 + i5;
                    arrayList2.add(Integer.valueOf(i7));
                    arrayList2.add(Integer.valueOf(i4 + i6));
                    arrayList2.add(Integer.valueOf(i7 + 24 + 1));
                }
                if (!z2) {
                    int i8 = i4 + i6;
                    arrayList2.add(Integer.valueOf(i8));
                    arrayList2.add(Integer.valueOf(i8 + 24 + 1));
                    arrayList2.add(Integer.valueOf(i5 + i4 + 24 + 1));
                }
                i5 = i6;
            }
            i4 += 25;
            i3++;
        }
        try {
            ModelRenderable modelRenderable = ModelRenderable.builder().setSource(RenderableDefinition.builder().setVertices(arrayList).setSubmeshes(Arrays.asList(RenderableDefinition.Submesh.builder().setTriangleIndices(arrayList2).setMaterial(material).build())).build()).build().get();
            if (modelRenderable != null) {
                return modelRenderable;
            }
            throw new AssertionError("Error creating renderable.");
        } catch (InterruptedException | ExecutionException e2) {
            throw new AssertionError("Error creating renderable.", e2);
        }
    }
}