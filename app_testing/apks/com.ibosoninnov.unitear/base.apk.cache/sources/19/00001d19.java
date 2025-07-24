package com.google.ar.sceneform.rendering;

import com.google.ar.sceneform.collision.Box;
import com.google.ar.sceneform.collision.CollisionShape;
import com.google.ar.sceneform.collision.Sphere;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.schemas.sceneform.SceneformBundleDef;
import com.google.ar.schemas.sceneform.SuggestedCollisionShapeDef;
import java.io.IOException;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class SceneformBundle {
    public static final float RCB_MAJOR_VERSION = 0.54f;
    public static final int RCB_MINOR_VERSION = 2;
    private static final char[] RCB_SIGNATURE = {'R', 'B', 'U', 'N'};
    private static final int SIGNATURE_OFFSET = 4;
    private static final String TAG = "SceneformBundle";

    /* loaded from: classes.dex */
    public static class VersionException extends Exception {
        public VersionException(String str) {
            super(str);
        }
    }

    public static boolean isSceneformBundle(ByteBuffer byteBuffer) {
        int i = 0;
        while (true) {
            char[] cArr = RCB_SIGNATURE;
            if (i >= cArr.length) {
                return true;
            }
            if (byteBuffer.get(i + 4) != cArr[i]) {
                return false;
            }
            i++;
        }
    }

    public static CollisionShape readCollisionGeometry(SceneformBundleDef sceneformBundleDef) {
        SuggestedCollisionShapeDef suggestedCollisionShape = sceneformBundleDef.suggestedCollisionShape();
        int type = suggestedCollisionShape.type();
        if (type == 0) {
            Box box = new Box();
            box.setCenter(new Vector3(suggestedCollisionShape.center().x(), suggestedCollisionShape.center().y(), suggestedCollisionShape.center().z()));
            box.setSize(new Vector3(suggestedCollisionShape.size().x(), suggestedCollisionShape.size().y(), suggestedCollisionShape.size().z()));
            return box;
        } else if (type == 1) {
            Sphere sphere = new Sphere();
            sphere.setCenter(new Vector3(suggestedCollisionShape.center().x(), suggestedCollisionShape.center().y(), suggestedCollisionShape.center().z()));
            sphere.setRadius(suggestedCollisionShape.size().x());
            return sphere;
        } else {
            throw new IOException("Invalid collisionCollisionGeometry type.");
        }
    }

    public static SceneformBundleDef tryLoadSceneformBundle(ByteBuffer byteBuffer) {
        if (isSceneformBundle(byteBuffer)) {
            byteBuffer.rewind();
            SceneformBundleDef rootAsSceneformBundleDef = SceneformBundleDef.getRootAsSceneformBundleDef(byteBuffer);
            float majorVersion = rootAsSceneformBundleDef.version().majorVersion();
            int minorVersion = rootAsSceneformBundleDef.version().minorVersion();
            if (0.54f >= rootAsSceneformBundleDef.version().majorVersion()) {
                return rootAsSceneformBundleDef;
            }
            throw new VersionException("Sceneform bundle (.sfb) version not supported, max version supported is 0.54.X. Version requested for loading is " + majorVersion + "." + minorVersion);
        }
        return null;
    }
}