package androidx.lifecycle;

import b.t.b;
import b.t.e;
import b.t.f;
import b.t.h;

/* loaded from: classes.dex */
public class ReflectiveGenericLifecycleObserver implements f {

    /* renamed from: a  reason: collision with root package name */
    public final Object f326a;

    /* renamed from: b  reason: collision with root package name */
    public final b.a f327b;

    public ReflectiveGenericLifecycleObserver(Object obj) {
        this.f326a = obj;
        this.f327b = b.f2565a.b(obj.getClass());
    }

    @Override // b.t.f
    public void e(h hVar, e.a aVar) {
        b.a aVar2 = this.f327b;
        Object obj = this.f326a;
        b.a.a(aVar2.f2568a.get(aVar), hVar, aVar, obj);
        b.a.a(aVar2.f2568a.get(e.a.ON_ANY), hVar, aVar, obj);
    }
}