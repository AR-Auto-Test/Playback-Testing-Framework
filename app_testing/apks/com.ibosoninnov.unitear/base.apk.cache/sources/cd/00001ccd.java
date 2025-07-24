package com.google.ar.sceneform.rendering;

import com.google.android.filament.Entity;
import com.google.android.filament.IndexBuffer;
import com.google.android.filament.VertexBuffer;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.RenderableInternalData;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes.dex */
public interface IRenderableInternalData {
    void buildInstanceData(Renderable renderable, @Entity int i);

    void dispose();

    List<String> getAnimationNames();

    Vector3 getCenterAabb();

    Vector3 getExtentsAabb();

    IndexBuffer getIndexBuffer();

    ArrayList<RenderableInternalData.MeshData> getMeshes();

    FloatBuffer getRawColorBuffer();

    IntBuffer getRawIndexBuffer();

    FloatBuffer getRawPositionBuffer();

    FloatBuffer getRawTangentsBuffer();

    FloatBuffer getRawUvBuffer();

    Vector3 getSizeAabb();

    Vector3 getTransformOffset();

    float getTransformScale();

    VertexBuffer getVertexBuffer();

    void setAnimationNames(List<String> list);

    void setCenterAabb(Vector3 vector3);

    void setExtentsAabb(Vector3 vector3);

    void setIndexBuffer(IndexBuffer indexBuffer);

    void setRawColorBuffer(FloatBuffer floatBuffer);

    void setRawIndexBuffer(IntBuffer intBuffer);

    void setRawPositionBuffer(FloatBuffer floatBuffer);

    void setRawTangentsBuffer(FloatBuffer floatBuffer);

    void setRawUvBuffer(FloatBuffer floatBuffer);

    void setTransformOffset(Vector3 vector3);

    void setTransformScale(float f2);

    void setVertexBuffer(VertexBuffer vertexBuffer);
}