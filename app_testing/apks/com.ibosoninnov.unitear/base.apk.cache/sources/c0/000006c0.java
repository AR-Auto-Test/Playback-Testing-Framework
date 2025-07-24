package c.a.a.z.j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/* compiled from: BaseAnimatableValue.java */
/* loaded from: classes.dex */
public abstract class n<V, O> implements m<V, O> {

    /* renamed from: a  reason: collision with root package name */
    public final List<c.a.a.d0.a<V>> f3301a;

    public n(V v) {
        this.f3301a = Collections.singletonList(new c.a.a.d0.a(v));
    }

    @Override // c.a.a.z.j.m
    public List<c.a.a.d0.a<V>> b() {
        return this.f3301a;
    }

    @Override // c.a.a.z.j.m
    public boolean c() {
        return this.f3301a.isEmpty() || (this.f3301a.size() == 1 && this.f3301a.get(0).d());
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (!this.f3301a.isEmpty()) {
            sb.append("values=");
            sb.append(Arrays.toString(this.f3301a.toArray()));
        }
        return sb.toString();
    }

    public n(List<c.a.a.d0.a<V>> list) {
        this.f3301a = list;
    }
}