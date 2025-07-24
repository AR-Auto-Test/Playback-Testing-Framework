package com.google.ar.sceneform.rendering;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.core.Plane;
import com.google.ar.core.TrackingState;
import com.google.ar.sceneform.common.TransformProvider;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.RenderableDefinition;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

/* loaded from: classes.dex */
public class PlaneVisualizer implements TransformProvider {
    private static final float FEATHER_LENGTH = 0.2f;
    private static final float FEATHER_SCALE = 0.2f;
    private static final String TAG = "PlaneVisualizer";
    private static final int VERTS_PER_BOUNDARY_VERT = 2;
    private final Plane plane;
    private RenderableInstance planeRenderableInstance;
    private RenderableDefinition.Submesh planeSubmesh;
    private final RenderableDefinition renderableDefinition;
    private final Renderer renderer;
    private RenderableDefinition.Submesh shadowSubmesh;
    private final ArrayList<Integer> triangleIndices;
    private final ArrayList<Vertex> vertices;
    private final Matrix planeMatrix = new Matrix();
    private boolean isPlaneAddedToScene = false;
    private boolean isEnabled = false;
    private boolean isShadowReceiver = false;
    private boolean isVisible = false;
    private ModelRenderable planeRenderable = null;

    public PlaneVisualizer(Plane plane, Renderer renderer) {
        ArrayList<Vertex> arrayList = new ArrayList<>();
        this.vertices = arrayList;
        this.triangleIndices = new ArrayList<>();
        this.plane = plane;
        this.renderer = renderer;
        this.renderableDefinition = RenderableDefinition.builder().setVertices(arrayList).build();
    }

    private void addPlaneToScene() {
        RenderableInstance renderableInstance;
        if (this.isPlaneAddedToScene || (renderableInstance = this.planeRenderableInstance) == null) {
            return;
        }
        this.renderer.addInstance(renderableInstance);
        this.isPlaneAddedToScene = true;
    }

    private void removePlaneFromScene() {
        RenderableInstance renderableInstance;
        if (!this.isPlaneAddedToScene || (renderableInstance = this.planeRenderableInstance) == null) {
            return;
        }
        this.renderer.removeInstance(renderableInstance);
        this.isPlaneAddedToScene = false;
    }

    private boolean updateRenderableDefinitionForPlane() {
        FloatBuffer polygon = this.plane.getPolygon();
        if (polygon == null) {
            return false;
        }
        polygon.rewind();
        int limit = polygon.limit() / 2;
        if (limit == 0) {
            return false;
        }
        this.vertices.clear();
        this.vertices.ensureCapacity(limit * 2);
        int i = limit - 2;
        this.triangleIndices.clear();
        this.triangleIndices.ensureCapacity((i * 3) + (limit * 6));
        Vector3 up = Vector3.up();
        while (polygon.hasRemaining()) {
            this.vertices.add(Vertex.builder().setPosition(new Vector3(polygon.get(), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, polygon.get())).setNormal(up).build());
        }
        polygon.rewind();
        while (polygon.hasRemaining()) {
            float f2 = polygon.get();
            float f3 = polygon.get();
            float hypot = (float) Math.hypot(f2, f3);
            float f4 = 0.8f;
            if (hypot != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                f4 = 1.0f - Math.min(0.2f / hypot, 0.2f);
            }
            this.vertices.add(Vertex.builder().setPosition(new Vector3(f2 * f4, 1.0f, f3 * f4)).setNormal(up).build());
        }
        short s = (short) limit;
        for (int i2 = 0; i2 < i; i2++) {
            this.triangleIndices.add(Integer.valueOf(s));
            int i3 = s + i2;
            this.triangleIndices.add(Integer.valueOf(i3 + 1));
            this.triangleIndices.add(Integer.valueOf(i3 + 2));
        }
        short s2 = 0;
        while (s2 < limit) {
            int i4 = 0 + s2;
            int i5 = s2 + 1;
            int i6 = i5 % limit;
            int i7 = 0 + i6;
            int i8 = s2 + s;
            this.triangleIndices.add(Integer.valueOf(i4));
            this.triangleIndices.add(Integer.valueOf(i7));
            this.triangleIndices.add(Integer.valueOf(i8));
            this.triangleIndices.add(Integer.valueOf(i8));
            this.triangleIndices.add(Integer.valueOf(i7));
            this.triangleIndices.add(Integer.valueOf(i6 + s));
            s2 = i5;
        }
        return true;
    }

    public Plane getPlane() {
        return this.plane;
    }

    @Override // com.google.ar.sceneform.common.TransformProvider
    public Matrix getWorldModelMatrix() {
        return this.planeMatrix;
    }

    public void release() {
        removePlaneFromScene();
        this.planeRenderable = null;
    }

    public void setEnabled(boolean z) {
        if (this.isEnabled != z) {
            this.isEnabled = z;
            updatePlane();
        }
    }

    public void setPlaneMaterial(Material material) {
        RenderableDefinition.Submesh submesh = this.planeSubmesh;
        if (submesh == null) {
            this.planeSubmesh = RenderableDefinition.Submesh.builder().setTriangleIndices(this.triangleIndices).setMaterial(material).build();
        } else {
            submesh.setMaterial(material);
        }
        if (this.planeRenderable != null) {
            updateRenderable();
        }
    }

    public void setShadowMaterial(Material material) {
        RenderableDefinition.Submesh submesh = this.shadowSubmesh;
        if (submesh == null) {
            this.shadowSubmesh = RenderableDefinition.Submesh.builder().setTriangleIndices(this.triangleIndices).setMaterial(material).build();
        } else {
            submesh.setMaterial(material);
        }
        if (this.planeRenderable != null) {
            updateRenderable();
        }
    }

    public void setShadowReceiver(boolean z) {
        if (this.isShadowReceiver != z) {
            this.isShadowReceiver = z;
            updatePlane();
        }
    }

    public void setVisible(boolean z) {
        if (this.isVisible != z) {
            this.isVisible = z;
            updatePlane();
        }
    }

    public void updatePlane() {
        if (this.isEnabled && (this.isVisible || this.isShadowReceiver)) {
            if (this.plane.getTrackingState() != TrackingState.TRACKING) {
                removePlaneFromScene();
                return;
            }
            this.plane.getCenterPose().toMatrix(this.planeMatrix.data, 0);
            if (!updateRenderableDefinitionForPlane()) {
                removePlaneFromScene();
                return;
            }
            updateRenderable();
            addPlaneToScene();
            return;
        }
        removePlaneFromScene();
    }

    public void updateRenderable() {
        RenderableDefinition.Submesh submesh;
        RenderableDefinition.Submesh submesh2;
        List<RenderableDefinition.Submesh> submeshes = this.renderableDefinition.getSubmeshes();
        submeshes.clear();
        if (this.isVisible && (submesh2 = this.planeSubmesh) != null) {
            submeshes.add(submesh2);
        }
        if (this.isShadowReceiver && (submesh = this.shadowSubmesh) != null) {
            submeshes.add(submesh);
        }
        if (submeshes.isEmpty()) {
            removePlaneFromScene();
            return;
        }
        ModelRenderable modelRenderable = this.planeRenderable;
        if (modelRenderable == null) {
            try {
                ModelRenderable modelRenderable2 = ModelRenderable.builder().setSource(this.renderableDefinition).build().get();
                this.planeRenderable = modelRenderable2;
                modelRenderable2.setShadowCaster(false);
                this.planeRenderableInstance = this.planeRenderable.createInstance(this);
            } catch (InterruptedException | ExecutionException unused) {
                throw new AssertionError("Unable to create plane renderable.");
            }
        } else {
            modelRenderable.updateFromDefinition(this.renderableDefinition);
        }
        if (this.planeRenderableInstance == null || submeshes.size() <= 1) {
            return;
        }
        this.planeRenderableInstance.setBlendOrderAt(0, 0);
        this.planeRenderableInstance.setBlendOrderAt(1, 1);
    }
}