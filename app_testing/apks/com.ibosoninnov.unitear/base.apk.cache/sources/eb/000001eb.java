package b.b.h;

import android.content.Context;
import android.content.ContextWrapper;

/* compiled from: TintContextWrapper.java */
/* loaded from: classes.dex */
public class v0 extends ContextWrapper {

    /* renamed from: a  reason: collision with root package name */
    public static final Object f933a = new Object();

    public static Context a(Context context) {
        if (!(context instanceof v0) && !(context.getResources() instanceof x0)) {
            context.getResources();
            int i = d1.f822a;
        }
        return context;
    }
}