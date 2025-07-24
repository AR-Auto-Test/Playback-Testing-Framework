package com.google.ar.core;

/* compiled from: Session.java */
/* loaded from: classes.dex */
public enum af {
    BASE_TRACKABLE(1095893248, Trackable.class),
    UNKNOWN_TO_JAVA(-1, null),
    PLANE(1095893249, Plane.class),
    POINT(1095893250, Point.class),
    AUGMENTED_IMAGE(1095893252, AugmentedImage.class),
    FACE(1095893253, AugmentedFace.class),
    DEPTH_POINT(1095893265, DepthPoint.class),
    INSTANT_PLACEMENT_POINT(1095893266, InstantPlacementPoint.class);
    
    public final int i;
    private final Class<?> k;

    af(int i, Class cls) {
        this.i = i;
        this.k = cls;
    }

    public static af a(Class<? extends Trackable> cls) {
        af[] values = values();
        for (int i = 0; i < 8; i++) {
            af afVar = values[i];
            Class<?> cls2 = afVar.k;
            if (cls2 != null && cls2.equals(cls)) {
                return afVar;
            }
        }
        return UNKNOWN_TO_JAVA;
    }
}