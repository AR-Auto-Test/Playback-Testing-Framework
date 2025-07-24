package c.c.a.m.w;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/* compiled from: ModelLoader.java */
/* loaded from: classes.dex */
public interface n<Model, Data> {

    /* compiled from: ModelLoader.java */
    /* loaded from: classes.dex */
    public static class a<Data> {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.m.m f3863a;

        /* renamed from: b  reason: collision with root package name */
        public final List<c.c.a.m.m> f3864b;

        /* renamed from: c  reason: collision with root package name */
        public final c.c.a.m.u.d<Data> f3865c;

        public a(c.c.a.m.m mVar, c.c.a.m.u.d<Data> dVar) {
            List<c.c.a.m.m> emptyList = Collections.emptyList();
            Objects.requireNonNull(mVar, "Argument must not be null");
            this.f3863a = mVar;
            Objects.requireNonNull(emptyList, "Argument must not be null");
            this.f3864b = emptyList;
            Objects.requireNonNull(dVar, "Argument must not be null");
            this.f3865c = dVar;
        }
    }

    boolean a(Model model);

    a<Data> b(Model model, int i, int i2, c.c.a.m.p pVar);
}