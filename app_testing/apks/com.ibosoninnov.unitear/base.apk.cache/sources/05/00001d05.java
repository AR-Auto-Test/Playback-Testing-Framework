package com.google.ar.sceneform.rendering;

import com.google.android.filament.IndexBuffer;
import com.google.android.filament.VertexBuffer;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.MathHelper;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.RenderableInternalData;
import com.google.ar.sceneform.rendering.Vertex;
import com.google.ar.sceneform.utilities.AndroidPreconditions;
import com.google.ar.sceneform.utilities.Preconditions;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

/* loaded from: classes.dex */
public class RenderableDefinition {
    private static final int BYTES_PER_FLOAT = 4;
    private static final int COLOR_SIZE = 4;
    private static final int POSITION_SIZE = 3;
    private static final int TANGENTS_SIZE = 4;
    private static final int UV_SIZE = 2;
    private static final Matrix scratchMatrix = new Matrix();
    private List<Submesh> submeshes;
    private List<Vertex> vertices;

    /* loaded from: classes.dex */
    public static final class Builder {
        private List<Submesh> submeshes = new ArrayList();
        private List<Vertex> vertices;

        public RenderableDefinition build() {
            return new RenderableDefinition(this);
        }

        public Builder setSubmeshes(List<Submesh> list) {
            this.submeshes = list;
            return this;
        }

        public Builder setVertices(List<Vertex> list) {
            this.vertices = list;
            return this;
        }
    }

    /* loaded from: classes.dex */
    public static class Submesh {
        private Material material;
        private String name;
        private List<Integer> triangleIndices;

        /* loaded from: classes.dex */
        public static final class Builder {
            private Material material;
            private String name;
            private List<Integer> triangleIndices;

            public Submesh build() {
                return new Submesh(this);
            }

            public Builder setMaterial(Material material) {
                this.material = material;
                return this;
            }

            public Builder setName(String str) {
                this.name = str;
                return this;
            }

            public Builder setTriangleIndices(List<Integer> list) {
                this.triangleIndices = list;
                return this;
            }
        }

        public static Builder builder() {
            return new Builder();
        }

        public Material getMaterial() {
            return this.material;
        }

        public String getName() {
            return this.name;
        }

        public List<Integer> getTriangleIndices() {
            return this.triangleIndices;
        }

        public void setMaterial(Material material) {
            this.material = material;
        }

        public void setName(String str) {
            this.name = str;
        }

        public void setTriangleIndices(List<Integer> list) {
            this.triangleIndices = list;
        }

        private Submesh(Builder builder) {
            this.triangleIndices = (List) Preconditions.checkNotNull(builder.triangleIndices);
            this.material = (Material) Preconditions.checkNotNull(builder.material);
            this.name = builder.name;
        }
    }

    private static void addColorToBuffer(Color color, FloatBuffer floatBuffer) {
        floatBuffer.put(color.r);
        floatBuffer.put(color.f5628g);
        floatBuffer.put(color.f5627b);
        floatBuffer.put(color.f5626a);
    }

    private static void addQuaternionToBuffer(Quaternion quaternion, FloatBuffer floatBuffer) {
        floatBuffer.put(quaternion.x);
        floatBuffer.put(quaternion.y);
        floatBuffer.put(quaternion.z);
        floatBuffer.put(quaternion.w);
    }

    private static void addUvToBuffer(Vertex.UvCoordinate uvCoordinate, FloatBuffer floatBuffer) {
        floatBuffer.put(uvCoordinate.x);
        floatBuffer.put(uvCoordinate.y);
    }

    private static void addVector3ToBuffer(Vector3 vector3, FloatBuffer floatBuffer) {
        floatBuffer.put(vector3.x);
        floatBuffer.put(vector3.y);
        floatBuffer.put(vector3.z);
    }

    private void applyDefinitionToDataIndexBuffer(IRenderableInternalData iRenderableInternalData) {
        int i = 0;
        for (int i2 = 0; i2 < this.submeshes.size(); i2++) {
            i += this.submeshes.get(i2).getTriangleIndices().size();
        }
        IntBuffer rawIndexBuffer = iRenderableInternalData.getRawIndexBuffer();
        if (rawIndexBuffer != null && rawIndexBuffer.capacity() >= i) {
            rawIndexBuffer.rewind();
        } else {
            rawIndexBuffer = IntBuffer.allocate(i);
            iRenderableInternalData.setRawIndexBuffer(rawIndexBuffer);
        }
        for (int i3 = 0; i3 < this.submeshes.size(); i3++) {
            List<Integer> triangleIndices = this.submeshes.get(i3).getTriangleIndices();
            for (int i4 = 0; i4 < triangleIndices.size(); i4++) {
                rawIndexBuffer.put(triangleIndices.get(i4).intValue());
            }
        }
        rawIndexBuffer.rewind();
        IndexBuffer indexBuffer = iRenderableInternalData.getIndexBuffer();
        IEngine engine = EngineInstance.getEngine();
        if (indexBuffer == null || indexBuffer.getIndexCount() < i) {
            if (indexBuffer != null) {
                engine.destroyIndexBuffer(indexBuffer);
            }
            indexBuffer = new IndexBuffer.Builder().indexCount(i).bufferType(IndexBuffer.Builder.IndexType.UINT).build(engine.getFilamentEngine());
            iRenderableInternalData.setIndexBuffer(indexBuffer);
        }
        indexBuffer.setBuffer(engine.getFilamentEngine(), rawIndexBuffer, 0, i);
    }

    private void applyDefinitionToDataVertexBuffer(IRenderableInternalData iRenderableInternalData) {
        boolean z;
        FloatBuffer floatBuffer;
        if (!this.vertices.isEmpty()) {
            int size = this.vertices.size();
            int i = 0;
            Vertex vertex = this.vertices.get(0);
            VertexBuffer.VertexAttribute vertexAttribute = VertexBuffer.VertexAttribute.POSITION;
            EnumSet of = EnumSet.of(vertexAttribute);
            if (vertex.getNormal() != null) {
                of.add(VertexBuffer.VertexAttribute.TANGENTS);
            }
            if (vertex.getUvCoordinate() != null) {
                of.add(VertexBuffer.VertexAttribute.UV0);
            }
            if (vertex.getColor() != null) {
                of.add(VertexBuffer.VertexAttribute.COLOR);
            }
            VertexBuffer vertexBuffer = iRenderableInternalData.getVertexBuffer();
            if (vertexBuffer != null) {
                EnumSet of2 = EnumSet.of(vertexAttribute);
                if (iRenderableInternalData.getRawTangentsBuffer() != null) {
                    of2.add(VertexBuffer.VertexAttribute.TANGENTS);
                }
                if (iRenderableInternalData.getRawUvBuffer() != null) {
                    of2.add(VertexBuffer.VertexAttribute.UV0);
                }
                if (iRenderableInternalData.getRawColorBuffer() != null) {
                    of2.add(VertexBuffer.VertexAttribute.COLOR);
                }
                z = !of2.equals(of) || vertexBuffer.getVertexCount() < size;
                if (z) {
                    EngineInstance.getEngine().destroyVertexBuffer(vertexBuffer);
                }
            } else {
                z = true;
            }
            if (z) {
                vertexBuffer = createVertexBuffer(size, of);
                iRenderableInternalData.setVertexBuffer(vertexBuffer);
            }
            FloatBuffer rawPositionBuffer = iRenderableInternalData.getRawPositionBuffer();
            if (rawPositionBuffer != null && rawPositionBuffer.capacity() >= size * 3) {
                rawPositionBuffer.rewind();
            } else {
                rawPositionBuffer = FloatBuffer.allocate(size * 3);
                iRenderableInternalData.setRawPositionBuffer(rawPositionBuffer);
            }
            FloatBuffer floatBuffer2 = rawPositionBuffer;
            FloatBuffer rawTangentsBuffer = iRenderableInternalData.getRawTangentsBuffer();
            if (of.contains(VertexBuffer.VertexAttribute.TANGENTS) && (rawTangentsBuffer == null || rawTangentsBuffer.capacity() < size * 4)) {
                rawTangentsBuffer = FloatBuffer.allocate(size * 4);
                iRenderableInternalData.setRawTangentsBuffer(rawTangentsBuffer);
            } else if (rawTangentsBuffer != null) {
                rawTangentsBuffer.rewind();
            }
            FloatBuffer rawUvBuffer = iRenderableInternalData.getRawUvBuffer();
            if (of.contains(VertexBuffer.VertexAttribute.UV0) && (rawUvBuffer == null || rawUvBuffer.capacity() < size * 2)) {
                rawUvBuffer = FloatBuffer.allocate(size * 2);
                iRenderableInternalData.setRawUvBuffer(rawUvBuffer);
            } else if (rawUvBuffer != null) {
                rawUvBuffer.rewind();
            }
            FloatBuffer floatBuffer3 = rawUvBuffer;
            FloatBuffer rawColorBuffer = iRenderableInternalData.getRawColorBuffer();
            if (!of.contains(VertexBuffer.VertexAttribute.COLOR) || (rawColorBuffer != null && rawColorBuffer.capacity() >= size * 4)) {
                if (rawColorBuffer != null) {
                    rawColorBuffer.rewind();
                }
                floatBuffer = rawColorBuffer;
            } else {
                floatBuffer = FloatBuffer.allocate(size * 4);
                iRenderableInternalData.setRawColorBuffer(floatBuffer);
            }
            Vector3 vector3 = new Vector3();
            Vector3 vector32 = new Vector3();
            Vector3 position = vertex.getPosition();
            vector3.set(position);
            vector32.set(position);
            for (int i2 = 0; i2 < this.vertices.size(); i2++) {
                Vertex vertex2 = this.vertices.get(i2);
                Vector3 position2 = vertex2.getPosition();
                vector3.set(Vector3.min(vector3, position2));
                vector32.set(Vector3.max(vector32, position2));
                addVector3ToBuffer(position2, floatBuffer2);
                if (rawTangentsBuffer != null) {
                    Vector3 normal = vertex2.getNormal();
                    if (normal != null) {
                        addQuaternionToBuffer(normalToTangent(normal), rawTangentsBuffer);
                    } else {
                        throw new IllegalArgumentException("Missing normal: If any Vertex in a RenderableDescription has a normal, all vertices must have one.");
                    }
                }
                if (floatBuffer3 != null) {
                    Vertex.UvCoordinate uvCoordinate = vertex2.getUvCoordinate();
                    if (uvCoordinate != null) {
                        addUvToBuffer(uvCoordinate, floatBuffer3);
                    } else {
                        throw new IllegalArgumentException("Missing UV Coordinate: If any Vertex in a RenderableDescription has a UV Coordinate, all vertices must have one.");
                    }
                }
                if (floatBuffer != null) {
                    Color color = vertex2.getColor();
                    if (color != null) {
                        addColorToBuffer(color, floatBuffer);
                    } else {
                        throw new IllegalArgumentException("Missing Color: If any Vertex in a RenderableDescription has a Color, all vertices must have one.");
                    }
                }
            }
            Vector3 scaled = Vector3.subtract(vector32, vector3).scaled(0.5f);
            Vector3 add = Vector3.add(vector3, scaled);
            iRenderableInternalData.setExtentsAabb(scaled);
            iRenderableInternalData.setCenterAabb(add);
            if (vertexBuffer != null) {
                IEngine engine = EngineInstance.getEngine();
                floatBuffer2.rewind();
                vertexBuffer.setBufferAt(engine.getFilamentEngine(), 0, floatBuffer2, 0, size * 3);
                if (rawTangentsBuffer != null) {
                    rawTangentsBuffer.rewind();
                    i = 1;
                    vertexBuffer.setBufferAt(engine.getFilamentEngine(), 1, rawTangentsBuffer, 0, size * 4);
                }
                if (floatBuffer3 != null) {
                    floatBuffer3.rewind();
                    i++;
                    vertexBuffer.setBufferAt(engine.getFilamentEngine(), i, floatBuffer3, 0, size * 2);
                }
                if (floatBuffer != null) {
                    floatBuffer.rewind();
                    vertexBuffer.setBufferAt(engine.getFilamentEngine(), i + 1, floatBuffer, 0, size * 4);
                    return;
                }
                return;
            }
            throw new AssertionError("VertexBuffer is null.");
        }
        throw new IllegalArgumentException("RenderableDescription must have at least one vertex.");
    }

    public static Builder builder() {
        return new Builder();
    }

    private static VertexBuffer createVertexBuffer(int i, EnumSet<VertexBuffer.VertexAttribute> enumSet) {
        int i2;
        VertexBuffer.Builder builder = new VertexBuffer.Builder();
        builder.vertexCount(i).bufferCount(enumSet.size());
        builder.attribute(VertexBuffer.VertexAttribute.POSITION, 0, VertexBuffer.AttributeType.FLOAT3, 0, 12);
        VertexBuffer.VertexAttribute vertexAttribute = VertexBuffer.VertexAttribute.TANGENTS;
        if (enumSet.contains(vertexAttribute)) {
            i2 = 1;
            builder.attribute(vertexAttribute, 1, VertexBuffer.AttributeType.FLOAT4, 0, 16);
        } else {
            i2 = 0;
        }
        VertexBuffer.VertexAttribute vertexAttribute2 = VertexBuffer.VertexAttribute.UV0;
        if (enumSet.contains(vertexAttribute2)) {
            i2++;
            builder.attribute(vertexAttribute2, i2, VertexBuffer.AttributeType.FLOAT2, 0, 8);
        }
        VertexBuffer.VertexAttribute vertexAttribute3 = VertexBuffer.VertexAttribute.COLOR;
        if (enumSet.contains(vertexAttribute3)) {
            builder.attribute(vertexAttribute3, i2 + 1, VertexBuffer.AttributeType.FLOAT4, 0, 16);
        }
        return builder.build(EngineInstance.getEngine().getFilamentEngine());
    }

    private static Quaternion normalToTangent(Vector3 vector3) {
        Vector3 normalized;
        Vector3 cross = Vector3.cross(Vector3.up(), vector3);
        if (MathHelper.almostEqualRelativeAndAbs(Vector3.dot(cross, cross), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            Vector3 normalized2 = Vector3.cross(vector3, Vector3.right()).normalized();
            normalized = normalized2;
            cross = Vector3.cross(normalized2, vector3).normalized();
        } else {
            cross.set(cross.normalized());
            normalized = Vector3.cross(vector3, cross).normalized();
        }
        Matrix matrix = scratchMatrix;
        float[] fArr = matrix.data;
        fArr[0] = cross.x;
        fArr[1] = cross.y;
        fArr[2] = cross.z;
        fArr[4] = normalized.x;
        fArr[5] = normalized.y;
        fArr[6] = normalized.z;
        fArr[8] = vector3.x;
        fArr[9] = vector3.y;
        fArr[10] = vector3.z;
        Quaternion quaternion = new Quaternion();
        matrix.extractQuaternion(quaternion);
        return quaternion;
    }

    public void applyDefinitionToData(IRenderableInternalData iRenderableInternalData, ArrayList<Material> arrayList, ArrayList<String> arrayList2) {
        RenderableInternalData.MeshData meshData;
        AndroidPreconditions.checkUiThread();
        applyDefinitionToDataIndexBuffer(iRenderableInternalData);
        applyDefinitionToDataVertexBuffer(iRenderableInternalData);
        arrayList.clear();
        arrayList2.clear();
        int i = 0;
        for (int i2 = 0; i2 < this.submeshes.size(); i2++) {
            Submesh submesh = this.submeshes.get(i2);
            if (i2 < iRenderableInternalData.getMeshes().size()) {
                meshData = iRenderableInternalData.getMeshes().get(i2);
            } else {
                meshData = new RenderableInternalData.MeshData();
                iRenderableInternalData.getMeshes().add(meshData);
            }
            meshData.indexStart = i;
            i += submesh.getTriangleIndices().size();
            meshData.indexEnd = i;
            arrayList.add(submesh.getMaterial());
            String name = submesh.getName();
            if (name == null) {
                name = "";
            }
            arrayList2.add(name);
        }
        while (iRenderableInternalData.getMeshes().size() > this.submeshes.size()) {
            iRenderableInternalData.getMeshes().remove(iRenderableInternalData.getMeshes().size() - 1);
        }
    }

    public List<Submesh> getSubmeshes() {
        return this.submeshes;
    }

    public List<Vertex> getVertices() {
        return this.vertices;
    }

    public void setSubmeshes(List<Submesh> list) {
        this.submeshes = list;
    }

    public void setVertices(List<Vertex> list) {
        this.vertices = list;
    }

    private RenderableDefinition(Builder builder) {
        this.vertices = (List) Preconditions.checkNotNull(builder.vertices);
        this.submeshes = (List) Preconditions.checkNotNull(builder.submeshes);
    }
}