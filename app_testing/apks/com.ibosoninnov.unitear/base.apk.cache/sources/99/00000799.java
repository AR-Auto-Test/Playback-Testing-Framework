package c.c.a.m.v;

import c.c.a.m.v.j;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/* compiled from: LoadPath.java */
/* loaded from: classes.dex */
public class u<Data, ResourceType, Transcode> {

    /* renamed from: a  reason: collision with root package name */
    public final b.j.i.d<List<Throwable>> f3797a;

    /* renamed from: b  reason: collision with root package name */
    public final List<? extends j<Data, ResourceType, Transcode>> f3798b;

    /* renamed from: c  reason: collision with root package name */
    public final String f3799c;

    public u(Class<Data> cls, Class<ResourceType> cls2, Class<Transcode> cls3, List<j<Data, ResourceType, Transcode>> list, b.j.i.d<List<Throwable>> dVar) {
        this.f3797a = dVar;
        if (!list.isEmpty()) {
            this.f3798b = list;
            StringBuilder x = c.b.a.a.a.x("Failed LoadPath{");
            x.append(cls.getSimpleName());
            x.append("->");
            x.append(cls2.getSimpleName());
            x.append("->");
            x.append(cls3.getSimpleName());
            x.append("}");
            this.f3799c = x.toString();
            return;
        }
        throw new IllegalArgumentException("Must not be empty.");
    }

    public w<Transcode> a(c.c.a.m.u.e<Data> eVar, c.c.a.m.p pVar, int i, int i2, j.a<ResourceType> aVar) {
        List<Throwable> b2 = this.f3797a.b();
        Objects.requireNonNull(b2, "Argument must not be null");
        List<Throwable> list = b2;
        try {
            int size = this.f3798b.size();
            w<Transcode> wVar = null;
            for (int i3 = 0; i3 < size; i3++) {
                try {
                    wVar = this.f3798b.get(i3).a(eVar, i, i2, pVar, aVar);
                } catch (r e2) {
                    list.add(e2);
                }
                if (wVar != null) {
                    break;
                }
            }
            if (wVar != null) {
                return wVar;
            }
            throw new r(this.f3799c, new ArrayList(list));
        } finally {
            this.f3797a.a(list);
        }
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("LoadPath{decodePaths=");
        x.append(Arrays.toString(this.f3798b.toArray()));
        x.append('}');
        return x.toString();
    }
}