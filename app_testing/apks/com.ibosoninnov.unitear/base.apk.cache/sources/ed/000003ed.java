package b.h.b.i;

import b.h.b.i.l.n;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

/* compiled from: ConstraintAnchor.java */
/* loaded from: classes.dex */
public class c {

    /* renamed from: b  reason: collision with root package name */
    public int f1861b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f1862c;

    /* renamed from: d  reason: collision with root package name */
    public final d f1863d;

    /* renamed from: e  reason: collision with root package name */
    public final a f1864e;

    /* renamed from: f  reason: collision with root package name */
    public c f1865f;
    public b.h.b.h i;

    /* renamed from: a  reason: collision with root package name */
    public HashSet<c> f1860a = null;

    /* renamed from: g  reason: collision with root package name */
    public int f1866g = 0;

    /* renamed from: h  reason: collision with root package name */
    public int f1867h = -1;

    /* compiled from: ConstraintAnchor.java */
    /* loaded from: classes.dex */
    public enum a {
        NONE,
        LEFT,
        TOP,
        RIGHT,
        BOTTOM,
        BASELINE,
        CENTER,
        CENTER_X,
        CENTER_Y
    }

    public c(d dVar, a aVar) {
        this.f1863d = dVar;
        this.f1864e = aVar;
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    /* JADX WARN: Code restructure failed: missing block: B:14:0x0022, code lost:
        if (r6.f1863d.y == false) goto L15;
     */
    /* JADX WARN: Code restructure failed: missing block: B:22:0x003c, code lost:
        if (r4 != r10) goto L18;
     */
    /* JADX WARN: Code restructure failed: missing block: B:35:0x0056, code lost:
        if (r4 != r10) goto L15;
     */
    /* JADX WARN: Code restructure failed: missing block: B:48:0x006f, code lost:
        if (r4 != r2) goto L15;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean a(c cVar, int i, int i2, boolean z) {
        boolean z2;
        if (cVar == null) {
            h();
            return true;
        }
        if (!z) {
            a aVar = a.CENTER_Y;
            a aVar2 = a.CENTER_X;
            a aVar3 = a.BASELINE;
            a aVar4 = cVar.f1864e;
            a aVar5 = this.f1864e;
            if (aVar4 == aVar5) {
                if (aVar5 == aVar3) {
                    if (cVar.f1863d.y) {
                    }
                    z2 = false;
                }
                z2 = true;
            } else {
                switch (aVar5.ordinal()) {
                    case 0:
                    case 5:
                    case 7:
                    case 8:
                        z2 = false;
                        break;
                    case 1:
                    case 3:
                        z2 = aVar4 == a.LEFT || aVar4 == a.RIGHT;
                        if (cVar.f1863d instanceof f) {
                            if (!z2) {
                            }
                            z2 = true;
                            break;
                        }
                        break;
                    case 2:
                    case 4:
                        boolean z3 = aVar4 == a.TOP || aVar4 == a.BOTTOM;
                        if (!(cVar.f1863d instanceof f)) {
                            z2 = z3;
                            break;
                        } else {
                            if (!z3) {
                            }
                            z2 = true;
                            break;
                        }
                        break;
                    case 6:
                        if (aVar4 != aVar3) {
                            if (aVar4 != aVar2) {
                            }
                        }
                        z2 = false;
                        break;
                    default:
                        throw new AssertionError(this.f1864e.name());
                }
            }
            if (!z2) {
                return false;
            }
        }
        this.f1865f = cVar;
        if (cVar.f1860a == null) {
            cVar.f1860a = new HashSet<>();
        }
        HashSet<c> hashSet = this.f1865f.f1860a;
        if (hashSet != null) {
            hashSet.add(this);
        }
        if (i > 0) {
            this.f1866g = i;
        } else {
            this.f1866g = 0;
        }
        this.f1867h = i2;
        return true;
    }

    public void b(int i, ArrayList<n> arrayList, n nVar) {
        HashSet<c> hashSet = this.f1860a;
        if (hashSet != null) {
            Iterator<c> it = hashSet.iterator();
            while (it.hasNext()) {
                b.e.a.b(it.next().f1863d, i, arrayList, nVar);
            }
        }
    }

    public int c() {
        if (this.f1862c) {
            return this.f1861b;
        }
        return 0;
    }

    public int d() {
        c cVar;
        if (this.f1863d.c0 == 8) {
            return 0;
        }
        int i = this.f1867h;
        return (i <= -1 || (cVar = this.f1865f) == null || cVar.f1863d.c0 != 8) ? this.f1866g : i;
    }

    public boolean e() {
        c cVar;
        HashSet<c> hashSet = this.f1860a;
        if (hashSet == null) {
            return false;
        }
        Iterator<c> it = hashSet.iterator();
        while (it.hasNext()) {
            c next = it.next();
            switch (next.f1864e.ordinal()) {
                case 0:
                case 5:
                case 6:
                case 7:
                case 8:
                    cVar = null;
                    break;
                case 1:
                    cVar = next.f1863d.F;
                    break;
                case 2:
                    cVar = next.f1863d.G;
                    break;
                case 3:
                    cVar = next.f1863d.D;
                    break;
                case 4:
                    cVar = next.f1863d.E;
                    break;
                default:
                    throw new AssertionError(next.f1864e.name());
            }
            if (cVar.g()) {
                return true;
            }
        }
        return false;
    }

    public boolean f() {
        HashSet<c> hashSet = this.f1860a;
        return hashSet != null && hashSet.size() > 0;
    }

    public boolean g() {
        return this.f1865f != null;
    }

    public void h() {
        HashSet<c> hashSet;
        c cVar = this.f1865f;
        if (cVar != null && (hashSet = cVar.f1860a) != null) {
            hashSet.remove(this);
            if (this.f1865f.f1860a.size() == 0) {
                this.f1865f.f1860a = null;
            }
        }
        this.f1860a = null;
        this.f1865f = null;
        this.f1866g = 0;
        this.f1867h = -1;
        this.f1862c = false;
        this.f1861b = 0;
    }

    public void i() {
        b.h.b.h hVar = this.i;
        if (hVar == null) {
            this.i = new b.h.b.h(1);
        } else {
            hVar.c();
        }
    }

    public void j(int i) {
        this.f1861b = i;
        this.f1862c = true;
    }

    public String toString() {
        return this.f1863d.d0 + ":" + this.f1864e.toString();
    }
}