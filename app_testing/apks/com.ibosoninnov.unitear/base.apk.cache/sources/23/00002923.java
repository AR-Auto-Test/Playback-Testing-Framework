package com.google.mediapipe.glutil;

import java.nio.FloatBuffer;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/glutil/CommonShaders.class */
public class CommonShaders {
    public static final String VERTEX_SHADER = "uniform mat4 texture_transform;\nattribute vec4 position;\nattribute mediump vec4 texture_coordinate;\nvarying mediump vec2 sample_coordinate;\n\nvoid main() {\n  gl_Position = position;\n  sample_coordinate = (texture_transform * texture_coordinate).xy;\n}";
    public static final String FRAGMENT_SHADER = "varying mediump vec2 sample_coordinate;\nuniform sampler2D video_frame;\n\nvoid main() {\n  gl_FragColor = texture2D(video_frame, sample_coordinate);\n}";
    public static final String FRAGMENT_SHADER_EXTERNAL = "#extension GL_OES_EGL_image_external : require\nvarying mediump vec2 sample_coordinate;\nuniform samplerExternalOES video_frame;\n\nvoid main() {\n  gl_FragColor = texture2D(video_frame, sample_coordinate);\n}";
    public static final FloatBuffer SQUARE_VERTICES = ShaderUtil.floatBuffer(-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    public static final FloatBuffer ROTATED_SQUARE_VERTICES = ShaderUtil.floatBuffer(-1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f);
}