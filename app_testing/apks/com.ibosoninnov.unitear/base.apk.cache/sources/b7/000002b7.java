package b.d.a.e.y1.o;

import android.util.Size;
import android.view.Surface;
import b.d.a.e.y1.o.b;
import java.util.List;
import java.util.Objects;

/* compiled from: OutputConfigurationCompatBaseImpl.java */
/* loaded from: classes.dex */
public class f implements b.a {

    /* renamed from: a  reason: collision with root package name */
    public final Object f1333a;

    /* compiled from: OutputConfigurationCompatBaseImpl.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final List<Surface> f1334a;

        /* renamed from: b  reason: collision with root package name */
        public final Size f1335b;

        /* renamed from: c  reason: collision with root package name */
        public final int f1336c;

        /* renamed from: d  reason: collision with root package name */
        public final int f1337d;

        public boolean equals(Object obj) {
            if (obj instanceof a) {
                a aVar = (a) obj;
                if (this.f1335b.equals(aVar.f1335b) && this.f1336c == aVar.f1336c && this.f1337d == aVar.f1337d && Objects.equals(null, null)) {
                    int min = Math.min(this.f1334a.size(), aVar.f1334a.size());
                    for (int i = 0; i < min; i++) {
                        if (this.f1334a.get(i) != aVar.f1334a.get(i)) {
                            return false;
                        }
                    }
                    return true;
                }
                return false;
            }
            return false;
        }

        public int hashCode() {
            int hashCode = this.f1334a.hashCode() ^ 31;
            int i = this.f1337d ^ ((hashCode << 5) - hashCode);
            int hashCode2 = this.f1335b.hashCode() ^ ((i << 5) - i);
            int i2 = this.f1336c ^ ((hashCode2 << 5) - hashCode2);
            int i3 = ((i2 << 5) - i2) ^ 0;
            return ((i3 << 5) - i3) ^ 0;
        }
    }

    public f(Object obj) {
        this.f1333a = obj;
    }

    @Override // b.d.a.e.y1.o.b.a
    public Surface a() {
        List<Surface> list = ((a) this.f1333a).f1334a;
        if (list.size() == 0) {
            return null;
        }
        return list.get(0);
    }

    @Override // b.d.a.e.y1.o.b.a
    public String b() {
        Objects.requireNonNull((a) this.f1333a);
        return null;
    }

    @Override // b.d.a.e.y1.o.b.a
    public Object c() {
        return null;
    }

    public boolean equals(Object obj) {
        if (obj instanceof f) {
            return Objects.equals(this.f1333a, ((f) obj).f1333a);
        }
        return false;
    }

    public int hashCode() {
        return this.f1333a.hashCode();
    }
}