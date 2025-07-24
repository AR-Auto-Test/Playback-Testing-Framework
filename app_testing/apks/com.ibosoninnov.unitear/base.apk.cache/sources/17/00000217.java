package b.d.a.d;

import b.d.b.d1.s0;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* compiled from: CameraEventCallbacks.java */
/* loaded from: classes.dex */
public final class c extends s0<b> {

    /* compiled from: CameraEventCallbacks.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final List<b> f1012a = new ArrayList();

        public a(List<b> list) {
            for (b bVar : list) {
                this.f1012a.add(bVar);
            }
        }
    }

    public c(b... bVarArr) {
        this.f1588a.addAll(Arrays.asList(bVarArr));
    }

    public static c d() {
        return new c(new b[0]);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // b.d.b.d1.s0
    /* renamed from: a */
    public s0<b> clone() {
        c d2 = d();
        d2.f1588a.addAll(b());
        return d2;
    }

    public a c() {
        return new a(b());
    }
}