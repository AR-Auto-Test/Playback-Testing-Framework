package c.a.a.z.k;

/* compiled from: MergePaths.java */
/* loaded from: classes.dex */
public class g implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3329a;

    /* renamed from: b  reason: collision with root package name */
    public final a f3330b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f3331c;

    /* compiled from: MergePaths.java */
    /* loaded from: classes.dex */
    public enum a {
        MERGE,
        ADD,
        SUBTRACT,
        INTERSECT,
        EXCLUDE_INTERSECTIONS
    }

    public g(String str, a aVar, boolean z) {
        this.f3329a = str;
        this.f3330b = aVar;
        this.f3331c = z;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        if (!jVar.o) {
            c.a.a.c0.c.b("Animation contains merge paths but they are disabled.");
            return null;
        }
        return new c.a.a.x.b.l(this);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("MergePaths{mode=");
        x.append(this.f3330b);
        x.append('}');
        return x.toString();
    }
}