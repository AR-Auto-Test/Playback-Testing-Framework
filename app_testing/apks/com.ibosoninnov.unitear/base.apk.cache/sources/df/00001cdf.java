package com.google.ar.sceneform.rendering;

import android.util.Log;
import com.google.android.filament.TextureSampler;
import com.google.ar.schemas.lull.ModelInstanceDef;
import com.google.ar.schemas.lull.TextureDef;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public class LullModel {
    private static final String TAG = "com.google.ar.sceneform.rendering.LullModel";
    public static final TextureSampler.WrapMode[] fromLullWrapMode;

    static {
        TextureSampler.WrapMode wrapMode = TextureSampler.WrapMode.CLAMP_TO_EDGE;
        fromLullWrapMode = new TextureSampler.WrapMode[]{wrapMode, wrapMode, TextureSampler.WrapMode.MIRRORED_REPEAT, wrapMode, TextureSampler.WrapMode.REPEAT};
    }

    public static TextureSampler.MagFilter fromLullToMagFilter(TextureDef textureDef) {
        int magFilter = textureDef.magFilter();
        if (magFilter != 0) {
            if (magFilter != 1) {
                String str = TAG;
                Log.e(str, textureDef.name() + ": Sampler has unknown mag filter");
                return TextureSampler.MagFilter.NEAREST;
            }
            return TextureSampler.MagFilter.LINEAR;
        }
        return TextureSampler.MagFilter.NEAREST;
    }

    public static TextureSampler.MinFilter fromLullToMinFilter(TextureDef textureDef) {
        int minFilter = textureDef.minFilter();
        if (minFilter != 0) {
            if (minFilter != 1) {
                if (minFilter != 2) {
                    if (minFilter != 3) {
                        if (minFilter != 4) {
                            if (minFilter != 5) {
                                String str = TAG;
                                Log.e(str, textureDef.name() + ": Sampler has unknown min filter");
                                return TextureSampler.MinFilter.NEAREST;
                            }
                            return TextureSampler.MinFilter.LINEAR_MIPMAP_LINEAR;
                        }
                        return TextureSampler.MinFilter.NEAREST_MIPMAP_LINEAR;
                    }
                    return TextureSampler.MinFilter.LINEAR_MIPMAP_NEAREST;
                }
                return TextureSampler.MinFilter.NEAREST_MIPMAP_NEAREST;
            }
            return TextureSampler.MinFilter.LINEAR;
        }
        return TextureSampler.MinFilter.NEAREST;
    }

    public static int getByteCountPerVertex(ModelInstanceDef modelInstanceDef) {
        int vertexAttributesLength = modelInstanceDef.vertexAttributesLength();
        int i = 0;
        for (int i2 = 0; i2 < vertexAttributesLength; i2++) {
            switch (modelInstanceDef.vertexAttributes(i2).type()) {
                case 1:
                case 5:
                case 7:
                    i += 4;
                    break;
                case 2:
                case 6:
                    i += 8;
                    break;
                case 3:
                    i += 12;
                    break;
                case 4:
                    i += 16;
                    break;
            }
        }
        return i;
    }

    public static boolean isLullModel(ByteBuffer byteBuffer) {
        return byteBuffer.limit() > 4 && byteBuffer.get(0) < 32 && byteBuffer.get(1) == 0 && byteBuffer.get(2) == 0;
    }
}