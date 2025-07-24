package c.c.a.m.x.f;

import c.c.a.m.v.w;
import java.io.File;
import java.util.Objects;

/* compiled from: FileResource.java */
/* loaded from: classes.dex */
public class b implements w {

    /* renamed from: b  reason: collision with root package name */
    public final T f4025b;

    /* JADX DEBUG: Multi-variable search result rejected for r2v0, resolved type: java.io.File */
    /* JADX WARN: Multi-variable type inference failed */
    public b(File file) {
        Objects.requireNonNull(file, "Argument must not be null");
        this.f4025b = file;
    }

    @Override // c.c.a.m.v.w
    public void a() {
    }

    @Override // c.c.a.m.v.w
    public final /* bridge */ /* synthetic */ int c() {
        return 1;
    }

    @Override // c.c.a.m.v.w
    public Class d() {
        return this.f4025b.getClass();
    }

    @Override // c.c.a.m.v.w
    public final Object get() {
        return this.f4025b;
    }
}