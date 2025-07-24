package com.google.vr.dynamite.client;

/* compiled from: D8$$SyntheticClass */
/* loaded from: classes2.dex */
public final /* synthetic */ class f {
    public static /* synthetic */ boolean a(Object obj, Object obj2) {
        return obj == obj2 || (obj != null && obj.equals(obj2));
    }

    public static /* synthetic */ int b(Object obj) {
        if (obj == null) {
            return 0;
        }
        return obj.hashCode();
    }
}