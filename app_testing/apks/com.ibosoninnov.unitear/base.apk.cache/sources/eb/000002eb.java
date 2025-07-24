package b.d.b.d1;

import com.google.common.util.concurrent.ListenableFuture;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/* compiled from: CameraRepository.java */
/* loaded from: classes.dex */
public final class b0 {

    /* renamed from: a  reason: collision with root package name */
    public final Object f1409a = new Object();

    /* renamed from: b  reason: collision with root package name */
    public final Map<String, a0> f1410b = new LinkedHashMap();

    /* renamed from: c  reason: collision with root package name */
    public final Set<a0> f1411c = new HashSet();

    /* renamed from: d  reason: collision with root package name */
    public ListenableFuture<Void> f1412d;

    /* renamed from: e  reason: collision with root package name */
    public b.g.a.b<Void> f1413e;

    public LinkedHashSet<a0> a() {
        LinkedHashSet<a0> linkedHashSet;
        synchronized (this.f1409a) {
            linkedHashSet = new LinkedHashSet<>(this.f1410b.values());
        }
        return linkedHashSet;
    }

    public void b(y yVar) {
        synchronized (this.f1409a) {
            try {
                try {
                    b.d.a.e.p0 p0Var = (b.d.a.e.p0) yVar;
                    for (String str : p0Var.a()) {
                        b.d.b.u0.a("CameraRepository", "Added camera: " + str, null);
                        this.f1410b.put(str, p0Var.b(str));
                    }
                } catch (b.d.b.k0 e2) {
                    throw new b.d.b.t0(e2);
                }
            } catch (Throwable th) {
                throw th;
            }
        }
    }
}