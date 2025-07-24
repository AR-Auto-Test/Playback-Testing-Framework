package com.google.ar.sceneform.rendering;

import android.content.Context;
import com.google.ar.sceneform.utilities.LoadHelper;

/* loaded from: classes.dex */
public final class RenderingResources {

    /* renamed from: com.google.ar.sceneform.rendering.RenderingResources$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource;

        static {
            Resource.values();
            int[] iArr = new int[9];
            $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource = iArr;
            try {
                iArr[Resource.CAMERA_MATERIAL.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.OPAQUE_COLORED_MATERIAL.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.TRANSPARENT_COLORED_MATERIAL.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.OPAQUE_TEXTURED_MATERIAL.ordinal()] = 4;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.TRANSPARENT_TEXTURED_MATERIAL.ordinal()] = 5;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.PLANE_SHADOW_MATERIAL.ordinal()] = 6;
            } catch (NoSuchFieldError unused6) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.PLANE_MATERIAL.ordinal()] = 7;
            } catch (NoSuchFieldError unused7) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.PLANE.ordinal()] = 8;
            } catch (NoSuchFieldError unused8) {
            }
            try {
                $SwitchMap$com$google$ar$sceneform$rendering$RenderingResources$Resource[Resource.VIEW_RENDERABLE_MATERIAL.ordinal()] = 9;
            } catch (NoSuchFieldError unused9) {
            }
        }
    }

    /* loaded from: classes.dex */
    public enum Resource {
        CAMERA_MATERIAL,
        OPAQUE_COLORED_MATERIAL,
        TRANSPARENT_COLORED_MATERIAL,
        OPAQUE_TEXTURED_MATERIAL,
        TRANSPARENT_TEXTURED_MATERIAL,
        PLANE_SHADOW_MATERIAL,
        PLANE_MATERIAL,
        PLANE,
        VIEW_RENDERABLE_MATERIAL
    }

    private RenderingResources() {
    }

    private static int GetMaterialFactoryBlazeResource(Resource resource) {
        return 0;
    }

    private static int GetSceneformBlazeResource(Resource resource) {
        return 0;
    }

    public static int GetSceneformResource(Context context, Resource resource) {
        int GetSceneformBlazeResource = GetSceneformBlazeResource(resource);
        return GetSceneformBlazeResource != 0 ? GetSceneformBlazeResource : GetSceneformSourceResource(context, resource);
    }

    private static int GetSceneformSourceResource(Context context, Resource resource) {
        switch (resource.ordinal()) {
            case 0:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_camera_material");
            case 1:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_opaque_colored_material");
            case 2:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_transparent_colored_material");
            case 3:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_opaque_textured_material");
            case 4:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_transparent_textured_material");
            case 5:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_plane_shadow_material");
            case 6:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_plane_material");
            case 7:
                return LoadHelper.drawableResourceNameToIdentifier(context, "sceneform_plane");
            case 8:
                return LoadHelper.rawResourceNameToIdentifier(context, "sceneform_view_material");
            default:
                return 0;
        }
    }

    private static int GetViewRenderableBlazeResource(Resource resource) {
        return 0;
    }
}