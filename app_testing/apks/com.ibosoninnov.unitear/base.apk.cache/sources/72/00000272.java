package b.d.a.e;

import android.content.Context;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/* compiled from: Camera2DeviceSurfaceManager.java */
/* loaded from: classes.dex */
public final class v0 implements b.d.b.d1.x {

    /* renamed from: a  reason: collision with root package name */
    public final Map<String, o1> f1209a;

    /* renamed from: b  reason: collision with root package name */
    public final m0 f1210b;

    public v0(Context context, Object obj, Set<String> set) {
        b.d.a.e.y1.k a2;
        a aVar = a.f1013a;
        this.f1209a = new HashMap();
        this.f1210b = aVar;
        if (obj instanceof b.d.a.e.y1.k) {
            a2 = (b.d.a.e.y1.k) obj;
        } else {
            a2 = b.d.a.e.y1.k.a(context, b.d.b.d1.k1.a.a());
        }
        Objects.requireNonNull(context);
        for (String str : set) {
            this.f1209a.put(str, new o1(context, str, a2, this.f1210b));
        }
    }
}