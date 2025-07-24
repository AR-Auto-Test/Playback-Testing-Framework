package com.google.mediapipe.glutil;

import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import org.opencv.calib3d.Calib3d;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/glutil/ExternalTextureRenderer.class */
public class ExternalTextureRenderer {
    private static final FloatBuffer TEXTURE_VERTICES = ShaderUtil.floatBuffer(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, 1.0f, 1.0f);
    private static final FloatBuffer FLIPPED_TEXTURE_VERTICES = ShaderUtil.floatBuffer(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, 1.0f, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    private static final Vertex BOTTOM_LEFT = new Vertex(-1.0f, -1.0f);
    private static final Vertex BOTTOM_RIGHT = new Vertex(1.0f, -1.0f);
    private static final Vertex TOP_LEFT = new Vertex(-1.0f, 1.0f);
    private static final Vertex TOP_RIGHT = new Vertex(1.0f, 1.0f);
    private static final Vertex[] POSITION_VERTICIES = {BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT};
    private static final FloatBuffer POSITION_VERTICIES_0 = fb(POSITION_VERTICIES, 0, 1, 2, 3);
    private static final FloatBuffer POSITION_VERTICIES_90 = fb(POSITION_VERTICIES, 2, 0, 3, 1);
    private static final FloatBuffer POSITION_VERTICIES_180 = fb(POSITION_VERTICIES, 3, 2, 1, 0);
    private static final FloatBuffer POSITION_VERTICIES_270 = fb(POSITION_VERTICIES, 1, 3, 0, 2);
    private static final String TAG = "ExternalTextureRend";
    private static final int ATTRIB_POSITION = 1;
    private static final int ATTRIB_TEXTURE_COORDINATE = 2;
    private int frameUniform;
    private int textureTransformUniform;
    private boolean flipY;
    private int program = 0;
    private float[] textureTransformMatrix = new float[16];
    private int rotation = 0;

    public void setup() {
        Map<String, Integer> attributeLocations = new HashMap<>();
        attributeLocations.put("position", 1);
        attributeLocations.put("texture_coordinate", 2);
        this.program = ShaderUtil.createProgram("uniform mat4 texture_transform;\nattribute vec4 position;\nattribute mediump vec4 texture_coordinate;\nvarying mediump vec2 sample_coordinate;\n\nvoid main() {\n  gl_Position = position;\n  sample_coordinate = (texture_transform * texture_coordinate).xy;\n}", "#extension GL_OES_EGL_image_external : require\nvarying mediump vec2 sample_coordinate;\nuniform samplerExternalOES video_frame;\n\nvoid main() {\n  gl_FragColor = texture2D(video_frame, sample_coordinate);\n}", attributeLocations);
        this.frameUniform = GLES20.glGetUniformLocation(this.program, "video_frame");
        this.textureTransformUniform = GLES20.glGetUniformLocation(this.program, "texture_transform");
        ShaderUtil.checkGlError("glGetUniformLocation");
    }

    public void setFlipY(boolean flip) {
        this.flipY = flip;
    }

    public void setRotation(int rotation) {
        this.rotation = rotation;
    }

    public void render(SurfaceTexture surfaceTexture) {
        GLES20.glClear(Calib3d.CALIB_RATIONAL_MODEL);
        GLES20.glActiveTexture(33984);
        ShaderUtil.checkGlError("glActiveTexture");
        surfaceTexture.updateTexImage();
        surfaceTexture.getTransformMatrix(this.textureTransformMatrix);
        GLES20.glTexParameteri(36197, 10241, 9729);
        GLES20.glTexParameteri(36197, 10240, 9729);
        GLES20.glTexParameteri(36197, 10242, 33071);
        GLES20.glTexParameteri(36197, 10243, 33071);
        ShaderUtil.checkGlError("glTexParameteri");
        GLES20.glUseProgram(this.program);
        ShaderUtil.checkGlError("glUseProgram");
        GLES20.glUniform1i(this.frameUniform, 0);
        ShaderUtil.checkGlError("glUniform1i");
        GLES20.glUniformMatrix4fv(this.textureTransformUniform, 1, false, this.textureTransformMatrix, 0);
        ShaderUtil.checkGlError("glUniformMatrix4fv");
        GLES20.glEnableVertexAttribArray(1);
        GLES20.glVertexAttribPointer(1, 2, 5126, false, 0, (Buffer) getPositionVerticies());
        GLES20.glEnableVertexAttribArray(2);
        GLES20.glVertexAttribPointer(2, 2, 5126, false, 0, (Buffer) (this.flipY ? FLIPPED_TEXTURE_VERTICES : TEXTURE_VERTICES));
        ShaderUtil.checkGlError("program setup");
        GLES20.glDrawArrays(5, 0, 4);
        ShaderUtil.checkGlError("glDrawArrays");
        GLES20.glBindTexture(36197, 0);
        ShaderUtil.checkGlError("glBindTexture");
        GLES20.glFinish();
    }

    public void release() {
        GLES20.glDeleteProgram(this.program);
    }

    private FloatBuffer getPositionVerticies() {
        switch (this.rotation) {
            case 0:
            default:
                return POSITION_VERTICIES_0;
            case 1:
                return POSITION_VERTICIES_90;
            case 2:
                return POSITION_VERTICIES_180;
            case 3:
                return POSITION_VERTICIES_270;
        }
    }

    private static FloatBuffer fb(Vertex[] v, int i0, int i1, int i2, int i3) {
        return ShaderUtil.floatBuffer(v[i0].x, v[i0].y, v[i1].x, v[i1].y, v[i2].x, v[i2].y, v[i3].x, v[i3].y);
    }

    /* JADX INFO: Access modifiers changed from: private */
    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/glutil/ExternalTextureRenderer$Vertex.class */
    public static class Vertex {
        float x;
        float y;

        Vertex(float x, float y) {
            this.x = x;
            this.y = y;
        }
    }
}