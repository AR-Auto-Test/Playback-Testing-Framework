package b.u.a;

import b.f.i;
import b.j.b.d;
import b.t.h;
import b.t.m;
import b.t.n;
import b.t.s;
import b.t.u;
import b.t.v;
import b.t.x;
import b.t.y;
import java.io.FileDescriptor;
import java.io.PrintWriter;
import java.util.Objects;

/* compiled from: LoaderManagerImpl.java */
/* loaded from: classes.dex */
public class b extends b.u.a.a {

    /* renamed from: a  reason: collision with root package name */
    public final h f2605a;

    /* renamed from: b  reason: collision with root package name */
    public final C0049b f2606b;

    /* compiled from: LoaderManagerImpl.java */
    /* loaded from: classes.dex */
    public static class a<D> extends m<D> {
        @Override // androidx.lifecycle.LiveData
        public void e() {
            throw null;
        }

        @Override // androidx.lifecycle.LiveData
        public void f() {
            throw null;
        }

        /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: b.t.n<? super D> */
        /* JADX WARN: Multi-variable type inference failed */
        @Override // androidx.lifecycle.LiveData
        public void g(n<? super D> nVar) {
            super.g(nVar);
        }

        @Override // b.t.m, androidx.lifecycle.LiveData
        public void h(D d2) {
            super.h(d2);
        }

        public String toString() {
            StringBuilder sb = new StringBuilder(64);
            sb.append("LoaderInfo{");
            sb.append(Integer.toHexString(System.identityHashCode(this)));
            sb.append(" #");
            sb.append(0);
            sb.append(" : ");
            d.c(null, sb);
            sb.append("}}");
            return sb.toString();
        }
    }

    /* compiled from: LoaderManagerImpl.java */
    /* renamed from: b.u.a.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0049b extends s {

        /* renamed from: c  reason: collision with root package name */
        public static final u f2607c = new a();

        /* renamed from: d  reason: collision with root package name */
        public i<a> f2608d = new i<>(10);

        /* compiled from: LoaderManagerImpl.java */
        /* renamed from: b.u.a.b$b$a */
        /* loaded from: classes.dex */
        public static class a implements u {
            @Override // b.t.u
            public <T extends s> T a(Class<T> cls) {
                return new C0049b();
            }
        }

        @Override // b.t.s
        public void a() {
            if (this.f2608d.i() <= 0) {
                i<a> iVar = this.f2608d;
                int i = iVar.f1780f;
                Object[] objArr = iVar.f1779e;
                for (int i2 = 0; i2 < i; i2++) {
                    objArr[i2] = null;
                }
                iVar.f1780f = 0;
                iVar.f1777c = false;
                return;
            }
            Objects.requireNonNull(this.f2608d.j(0));
            throw null;
        }
    }

    public b(h hVar, y yVar) {
        s a2;
        this.f2605a = hVar;
        u uVar = C0049b.f2607c;
        String canonicalName = C0049b.class.getCanonicalName();
        if (canonicalName != null) {
            String q = c.b.a.a.a.q("androidx.lifecycle.ViewModelProvider.DefaultKey:", canonicalName);
            s sVar = yVar.f2604a.get(q);
            if (C0049b.class.isInstance(sVar)) {
                if (uVar instanceof x) {
                    ((x) uVar).b(sVar);
                }
            } else {
                if (uVar instanceof v) {
                    a2 = ((v) uVar).c(q, C0049b.class);
                } else {
                    a2 = ((C0049b.a) uVar).a(C0049b.class);
                }
                sVar = a2;
                s put = yVar.f2604a.put(q, sVar);
                if (put != null) {
                    put.a();
                }
            }
            this.f2606b = (C0049b) sVar;
            return;
        }
        throw new IllegalArgumentException("Local and anonymous classes can not be ViewModels");
    }

    @Override // b.u.a.a
    @Deprecated
    public void a(String str, FileDescriptor fileDescriptor, PrintWriter printWriter, String[] strArr) {
        C0049b c0049b = this.f2606b;
        if (c0049b.f2608d.i() > 0) {
            printWriter.print(str);
            printWriter.println("Loaders:");
            String str2 = str + "    ";
            if (c0049b.f2608d.i() <= 0) {
                return;
            }
            printWriter.print(str);
            printWriter.print("  #");
            printWriter.print(c0049b.f2608d.f(0));
            printWriter.print(": ");
            printWriter.println(c0049b.f2608d.j(0).toString());
            printWriter.print(str2);
            printWriter.print("mId=");
            printWriter.print(0);
            printWriter.print(" mArgs=");
            printWriter.println((Object) null);
            printWriter.print(str2);
            printWriter.print("mLoader=");
            printWriter.println((Object) null);
            throw null;
        }
    }

    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append("LoaderManager{");
        sb.append(Integer.toHexString(System.identityHashCode(this)));
        sb.append(" in ");
        d.c(this.f2605a, sb);
        sb.append("}}");
        return sb.toString();
    }
}