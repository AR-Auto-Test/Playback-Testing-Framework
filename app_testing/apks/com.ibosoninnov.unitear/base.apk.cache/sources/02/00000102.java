package androidx.recyclerview.widget;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.PointF;
import android.graphics.Rect;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityEvent;
import androidx.recyclerview.widget.RecyclerView;
import b.j.j.x.b;
import b.w.b.m;
import b.w.b.n;
import b.w.b.o;
import b.w.b.s;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;

/* loaded from: classes.dex */
public class StaggeredGridLayoutManager extends RecyclerView.o implements RecyclerView.z.b {

    /* renamed from: a  reason: collision with root package name */
    public int f451a;

    /* renamed from: b  reason: collision with root package name */
    public f[] f452b;

    /* renamed from: c  reason: collision with root package name */
    public s f453c;

    /* renamed from: d  reason: collision with root package name */
    public s f454d;

    /* renamed from: e  reason: collision with root package name */
    public int f455e;

    /* renamed from: f  reason: collision with root package name */
    public int f456f;

    /* renamed from: g  reason: collision with root package name */
    public final n f457g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f458h;
    public BitSet j;
    public boolean o;
    public boolean p;
    public e q;
    public int r;
    public int[] v;
    public boolean i = false;
    public int k = -1;
    public int l = Integer.MIN_VALUE;
    public d m = new d();
    public int n = 2;
    public final Rect s = new Rect();
    public final b t = new b();
    public boolean u = true;
    public final Runnable w = new a();

    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            StaggeredGridLayoutManager.this.b();
        }
    }

    /* loaded from: classes.dex */
    public class b {

        /* renamed from: a  reason: collision with root package name */
        public int f460a;

        /* renamed from: b  reason: collision with root package name */
        public int f461b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f462c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f463d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f464e;

        /* renamed from: f  reason: collision with root package name */
        public int[] f465f;

        public b() {
            b();
        }

        public void a() {
            this.f461b = this.f462c ? StaggeredGridLayoutManager.this.f453c.g() : StaggeredGridLayoutManager.this.f453c.k();
        }

        public void b() {
            this.f460a = -1;
            this.f461b = Integer.MIN_VALUE;
            this.f462c = false;
            this.f463d = false;
            this.f464e = false;
            int[] iArr = this.f465f;
            if (iArr != null) {
                Arrays.fill(iArr, -1);
            }
        }
    }

    /* loaded from: classes.dex */
    public static class c extends RecyclerView.p {

        /* renamed from: e  reason: collision with root package name */
        public f f467e;

        public c(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
        }

        public c(int i, int i2) {
            super(i, i2);
        }

        public c(ViewGroup.MarginLayoutParams marginLayoutParams) {
            super(marginLayoutParams);
        }

        public c(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
        }
    }

    @SuppressLint({"BanParcelableUsage"})
    /* loaded from: classes.dex */
    public static class e implements Parcelable {
        public static final Parcelable.Creator<e> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public int f474b;

        /* renamed from: c  reason: collision with root package name */
        public int f475c;

        /* renamed from: d  reason: collision with root package name */
        public int f476d;

        /* renamed from: e  reason: collision with root package name */
        public int[] f477e;

        /* renamed from: f  reason: collision with root package name */
        public int f478f;

        /* renamed from: g  reason: collision with root package name */
        public int[] f479g;

        /* renamed from: h  reason: collision with root package name */
        public List<d.a> f480h;
        public boolean i;
        public boolean j;
        public boolean k;

        /* loaded from: classes.dex */
        public static class a implements Parcelable.Creator<e> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.Creator
            public e createFromParcel(Parcel parcel) {
                return new e(parcel);
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
            @Override // android.os.Parcelable.Creator
            public e[] newArray(int i) {
                return new e[i];
            }
        }

        public e() {
        }

        @Override // android.os.Parcelable
        public int describeContents() {
            return 0;
        }

        @Override // android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            parcel.writeInt(this.f474b);
            parcel.writeInt(this.f475c);
            parcel.writeInt(this.f476d);
            if (this.f476d > 0) {
                parcel.writeIntArray(this.f477e);
            }
            parcel.writeInt(this.f478f);
            if (this.f478f > 0) {
                parcel.writeIntArray(this.f479g);
            }
            parcel.writeInt(this.i ? 1 : 0);
            parcel.writeInt(this.j ? 1 : 0);
            parcel.writeInt(this.k ? 1 : 0);
            parcel.writeList(this.f480h);
        }

        public e(Parcel parcel) {
            this.f474b = parcel.readInt();
            this.f475c = parcel.readInt();
            int readInt = parcel.readInt();
            this.f476d = readInt;
            if (readInt > 0) {
                int[] iArr = new int[readInt];
                this.f477e = iArr;
                parcel.readIntArray(iArr);
            }
            int readInt2 = parcel.readInt();
            this.f478f = readInt2;
            if (readInt2 > 0) {
                int[] iArr2 = new int[readInt2];
                this.f479g = iArr2;
                parcel.readIntArray(iArr2);
            }
            this.i = parcel.readInt() == 1;
            this.j = parcel.readInt() == 1;
            this.k = parcel.readInt() == 1;
            this.f480h = parcel.readArrayList(d.a.class.getClassLoader());
        }

        public e(e eVar) {
            this.f476d = eVar.f476d;
            this.f474b = eVar.f474b;
            this.f475c = eVar.f475c;
            this.f477e = eVar.f477e;
            this.f478f = eVar.f478f;
            this.f479g = eVar.f479g;
            this.i = eVar.i;
            this.j = eVar.j;
            this.k = eVar.k;
            this.f480h = eVar.f480h;
        }
    }

    /* loaded from: classes.dex */
    public class f {

        /* renamed from: a  reason: collision with root package name */
        public ArrayList<View> f481a = new ArrayList<>();

        /* renamed from: b  reason: collision with root package name */
        public int f482b = Integer.MIN_VALUE;

        /* renamed from: c  reason: collision with root package name */
        public int f483c = Integer.MIN_VALUE;

        /* renamed from: d  reason: collision with root package name */
        public int f484d = 0;

        /* renamed from: e  reason: collision with root package name */
        public final int f485e;

        public f(int i) {
            this.f485e = i;
        }

        public void a(View view) {
            c j = j(view);
            j.f467e = this;
            this.f481a.add(view);
            this.f483c = Integer.MIN_VALUE;
            if (this.f481a.size() == 1) {
                this.f482b = Integer.MIN_VALUE;
            }
            if (j.c() || j.b()) {
                this.f484d = StaggeredGridLayoutManager.this.f453c.c(view) + this.f484d;
            }
        }

        public void b() {
            ArrayList<View> arrayList = this.f481a;
            View view = arrayList.get(arrayList.size() - 1);
            c j = j(view);
            this.f483c = StaggeredGridLayoutManager.this.f453c.b(view);
            Objects.requireNonNull(j);
        }

        public void c() {
            View view = this.f481a.get(0);
            c j = j(view);
            this.f482b = StaggeredGridLayoutManager.this.f453c.e(view);
            Objects.requireNonNull(j);
        }

        public void d() {
            this.f481a.clear();
            this.f482b = Integer.MIN_VALUE;
            this.f483c = Integer.MIN_VALUE;
            this.f484d = 0;
        }

        public int e() {
            if (StaggeredGridLayoutManager.this.f458h) {
                return g(this.f481a.size() - 1, -1, true);
            }
            return g(0, this.f481a.size(), true);
        }

        public int f() {
            if (StaggeredGridLayoutManager.this.f458h) {
                return g(0, this.f481a.size(), true);
            }
            return g(this.f481a.size() - 1, -1, true);
        }

        public int g(int i, int i2, boolean z) {
            int k = StaggeredGridLayoutManager.this.f453c.k();
            int g2 = StaggeredGridLayoutManager.this.f453c.g();
            int i3 = i2 > i ? 1 : -1;
            while (i != i2) {
                View view = this.f481a.get(i);
                int e2 = StaggeredGridLayoutManager.this.f453c.e(view);
                int b2 = StaggeredGridLayoutManager.this.f453c.b(view);
                boolean z2 = false;
                boolean z3 = !z ? e2 >= g2 : e2 > g2;
                if (!z ? b2 > k : b2 >= k) {
                    z2 = true;
                }
                if (z3 && z2 && (e2 < k || b2 > g2)) {
                    return StaggeredGridLayoutManager.this.getPosition(view);
                }
                i += i3;
            }
            return -1;
        }

        public int h(int i) {
            int i2 = this.f483c;
            if (i2 != Integer.MIN_VALUE) {
                return i2;
            }
            if (this.f481a.size() == 0) {
                return i;
            }
            b();
            return this.f483c;
        }

        public View i(int i, int i2) {
            View view = null;
            if (i2 == -1) {
                int size = this.f481a.size();
                int i3 = 0;
                while (i3 < size) {
                    View view2 = this.f481a.get(i3);
                    StaggeredGridLayoutManager staggeredGridLayoutManager = StaggeredGridLayoutManager.this;
                    if (staggeredGridLayoutManager.f458h && staggeredGridLayoutManager.getPosition(view2) <= i) {
                        break;
                    }
                    StaggeredGridLayoutManager staggeredGridLayoutManager2 = StaggeredGridLayoutManager.this;
                    if ((!staggeredGridLayoutManager2.f458h && staggeredGridLayoutManager2.getPosition(view2) >= i) || !view2.hasFocusable()) {
                        break;
                    }
                    i3++;
                    view = view2;
                }
            } else {
                int size2 = this.f481a.size() - 1;
                while (size2 >= 0) {
                    View view3 = this.f481a.get(size2);
                    StaggeredGridLayoutManager staggeredGridLayoutManager3 = StaggeredGridLayoutManager.this;
                    if (staggeredGridLayoutManager3.f458h && staggeredGridLayoutManager3.getPosition(view3) >= i) {
                        break;
                    }
                    StaggeredGridLayoutManager staggeredGridLayoutManager4 = StaggeredGridLayoutManager.this;
                    if ((!staggeredGridLayoutManager4.f458h && staggeredGridLayoutManager4.getPosition(view3) <= i) || !view3.hasFocusable()) {
                        break;
                    }
                    size2--;
                    view = view3;
                }
            }
            return view;
        }

        public c j(View view) {
            return (c) view.getLayoutParams();
        }

        public int k(int i) {
            int i2 = this.f482b;
            if (i2 != Integer.MIN_VALUE) {
                return i2;
            }
            if (this.f481a.size() == 0) {
                return i;
            }
            c();
            return this.f482b;
        }

        public void l() {
            int size = this.f481a.size();
            View remove = this.f481a.remove(size - 1);
            c j = j(remove);
            j.f467e = null;
            if (j.c() || j.b()) {
                this.f484d -= StaggeredGridLayoutManager.this.f453c.c(remove);
            }
            if (size == 1) {
                this.f482b = Integer.MIN_VALUE;
            }
            this.f483c = Integer.MIN_VALUE;
        }

        public void m() {
            View remove = this.f481a.remove(0);
            c j = j(remove);
            j.f467e = null;
            if (this.f481a.size() == 0) {
                this.f483c = Integer.MIN_VALUE;
            }
            if (j.c() || j.b()) {
                this.f484d -= StaggeredGridLayoutManager.this.f453c.c(remove);
            }
            this.f482b = Integer.MIN_VALUE;
        }

        public void n(View view) {
            c j = j(view);
            j.f467e = this;
            this.f481a.add(0, view);
            this.f482b = Integer.MIN_VALUE;
            if (this.f481a.size() == 1) {
                this.f483c = Integer.MIN_VALUE;
            }
            if (j.c() || j.b()) {
                this.f484d = StaggeredGridLayoutManager.this.f453c.c(view) + this.f484d;
            }
        }
    }

    public StaggeredGridLayoutManager(Context context, AttributeSet attributeSet, int i, int i2) {
        this.f451a = -1;
        this.f458h = false;
        RecyclerView.o.d properties = RecyclerView.o.getProperties(context, attributeSet, i, i2);
        int i3 = properties.f420a;
        if (i3 != 0 && i3 != 1) {
            throw new IllegalArgumentException("invalid orientation.");
        }
        assertNotInLayoutOrScroll(null);
        if (i3 != this.f455e) {
            this.f455e = i3;
            s sVar = this.f453c;
            this.f453c = this.f454d;
            this.f454d = sVar;
            requestLayout();
        }
        int i4 = properties.f421b;
        assertNotInLayoutOrScroll(null);
        if (i4 != this.f451a) {
            this.m.a();
            requestLayout();
            this.f451a = i4;
            this.j = new BitSet(this.f451a);
            this.f452b = new f[this.f451a];
            for (int i5 = 0; i5 < this.f451a; i5++) {
                this.f452b[i5] = new f(i5);
            }
            requestLayout();
        }
        boolean z = properties.f422c;
        assertNotInLayoutOrScroll(null);
        e eVar = this.q;
        if (eVar != null && eVar.i != z) {
            eVar.i = z;
        }
        this.f458h = z;
        requestLayout();
        this.f457g = new n();
        this.f453c = s.a(this, this.f455e);
        this.f454d = s.a(this, 1 - this.f455e);
    }

    public final int a(int i) {
        if (getChildCount() == 0) {
            return this.i ? 1 : -1;
        }
        return (i < h()) != this.i ? -1 : 1;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void assertNotInLayoutOrScroll(String str) {
        if (this.q == null) {
            super.assertNotInLayoutOrScroll(str);
        }
    }

    public boolean b() {
        int h2;
        if (getChildCount() != 0 && this.n != 0 && isAttachedToWindow()) {
            if (this.i) {
                h2 = i();
                h();
            } else {
                h2 = h();
                i();
            }
            if (h2 == 0 && m() != null) {
                this.m.a();
                requestSimpleAnimationsInNextLayout();
                requestLayout();
                return true;
            }
        }
        return false;
    }

    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r17v0, types: [androidx.recyclerview.widget.RecyclerView$o, androidx.recyclerview.widget.StaggeredGridLayoutManager] */
    /* JADX WARN: Type inference failed for: r1v19 */
    /* JADX WARN: Type inference failed for: r1v20, types: [int, boolean] */
    /* JADX WARN: Type inference failed for: r1v42 */
    /* JADX WARN: Type inference failed for: r4v12, types: [int] */
    /* JADX WARN: Type inference failed for: r4v15, types: [int] */
    /* JADX WARN: Type inference failed for: r4v20 */
    /* JADX WARN: Type inference failed for: r4v22 */
    public final int c(RecyclerView.v vVar, n nVar, RecyclerView.a0 a0Var) {
        int i;
        int i2;
        int k;
        int j;
        f fVar;
        ?? r1;
        int i3;
        int c2;
        int k2;
        int c3;
        boolean z;
        int i4;
        int i5;
        boolean z2 = false;
        this.j.set(0, this.f451a, true);
        if (this.f457g.i) {
            i2 = nVar.f2789e == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        } else {
            if (nVar.f2789e == 1) {
                i = nVar.f2791g + nVar.f2786b;
            } else {
                i = nVar.f2790f - nVar.f2786b;
            }
            i2 = i;
        }
        v(nVar.f2789e, i2);
        if (this.i) {
            k = this.f453c.g();
        } else {
            k = this.f453c.k();
        }
        int i6 = k;
        boolean z3 = false;
        while (true) {
            int i7 = nVar.f2787c;
            int i8 = -1;
            if (!((i7 < 0 || i7 >= a0Var.b()) ? z2 : true) || (!this.f457g.i && this.j.isEmpty())) {
                break;
            }
            View view = vVar.k(nVar.f2787c, z2, RecyclerView.FOREVER_NS).itemView;
            nVar.f2787c += nVar.f2788d;
            c cVar = (c) view.getLayoutParams();
            int a2 = cVar.a();
            int[] iArr = this.m.f468a;
            int i9 = (iArr == null || a2 >= iArr.length) ? -1 : iArr[a2];
            if (i9 == -1 ? true : z2) {
                if (p(nVar.f2789e)) {
                    i4 = this.f451a - 1;
                    i5 = -1;
                } else {
                    i8 = this.f451a;
                    i4 = z2;
                    i5 = 1;
                }
                f fVar2 = null;
                if (nVar.f2789e == 1) {
                    int k3 = this.f453c.k();
                    int i10 = Integer.MAX_VALUE;
                    for (int i11 = i4; i11 != i8; i11 += i5) {
                        f fVar3 = this.f452b[i11];
                        int h2 = fVar3.h(k3);
                        if (h2 < i10) {
                            i10 = h2;
                            fVar2 = fVar3;
                        }
                    }
                } else {
                    int g2 = this.f453c.g();
                    int i12 = Integer.MIN_VALUE;
                    for (int i13 = i4; i13 != i8; i13 += i5) {
                        f fVar4 = this.f452b[i13];
                        int k4 = fVar4.k(g2);
                        if (k4 > i12) {
                            fVar2 = fVar4;
                            i12 = k4;
                        }
                    }
                }
                fVar = fVar2;
                d dVar = this.m;
                dVar.b(a2);
                dVar.f468a[a2] = fVar.f485e;
            } else {
                fVar = this.f452b[i9];
            }
            f fVar5 = fVar;
            cVar.f467e = fVar5;
            if (nVar.f2789e == 1) {
                addView(view);
                r1 = 0;
            } else {
                r1 = 0;
                addView(view, 0);
            }
            if (this.f455e == 1) {
                n(view, RecyclerView.o.getChildMeasureSpec(this.f456f, getWidthMode(), r1, ((ViewGroup.MarginLayoutParams) cVar).width, r1), RecyclerView.o.getChildMeasureSpec(getHeight(), getHeightMode(), getPaddingBottom() + getPaddingTop(), ((ViewGroup.MarginLayoutParams) cVar).height, true), r1);
            } else {
                n(view, RecyclerView.o.getChildMeasureSpec(getWidth(), getWidthMode(), getPaddingRight() + getPaddingLeft(), ((ViewGroup.MarginLayoutParams) cVar).width, true), RecyclerView.o.getChildMeasureSpec(this.f456f, getHeightMode(), 0, ((ViewGroup.MarginLayoutParams) cVar).height, false), false);
            }
            if (nVar.f2789e == 1) {
                int h3 = fVar5.h(i6);
                c2 = h3;
                i3 = this.f453c.c(view) + h3;
            } else {
                int k5 = fVar5.k(i6);
                i3 = k5;
                c2 = k5 - this.f453c.c(view);
            }
            if (nVar.f2789e == 1) {
                cVar.f467e.a(view);
            } else {
                cVar.f467e.n(view);
            }
            if (isLayoutRTL() && this.f455e == 1) {
                c3 = this.f454d.g() - (((this.f451a - 1) - fVar5.f485e) * this.f456f);
                k2 = c3 - this.f454d.c(view);
            } else {
                k2 = this.f454d.k() + (fVar5.f485e * this.f456f);
                c3 = this.f454d.c(view) + k2;
            }
            int i14 = c3;
            int i15 = k2;
            if (this.f455e == 1) {
                layoutDecoratedWithMargins(view, i15, c2, i14, i3);
            } else {
                layoutDecoratedWithMargins(view, c2, i15, i3, i14);
            }
            x(fVar5, this.f457g.f2789e, i2);
            r(vVar, this.f457g);
            if (this.f457g.f2792h && view.hasFocusable()) {
                z = false;
                this.j.set(fVar5.f485e, false);
            } else {
                z = false;
            }
            z2 = z;
            z3 = true;
        }
        boolean z4 = z2;
        if (!z3) {
            r(vVar, this.f457g);
        }
        if (this.f457g.f2789e == -1) {
            j = this.f453c.k() - k(this.f453c.k());
        } else {
            j = j(this.f453c.g()) - this.f453c.g();
        }
        if (j > 0) {
            return Math.min(nVar.f2786b, j);
        }
        return z4 ? 1 : 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean canScrollHorizontally() {
        return this.f455e == 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean canScrollVertically() {
        return this.f455e == 1;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean checkLayoutParams(RecyclerView.p pVar) {
        return pVar instanceof c;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void collectAdjacentPrefetchPositions(int i, int i2, RecyclerView.a0 a0Var, RecyclerView.o.c cVar) {
        int h2;
        int i3;
        if (this.f455e != 0) {
            i = i2;
        }
        if (getChildCount() == 0 || i == 0) {
            return;
        }
        q(i, a0Var);
        int[] iArr = this.v;
        if (iArr == null || iArr.length < this.f451a) {
            this.v = new int[this.f451a];
        }
        int i4 = 0;
        for (int i5 = 0; i5 < this.f451a; i5++) {
            n nVar = this.f457g;
            if (nVar.f2788d == -1) {
                h2 = nVar.f2790f;
                i3 = this.f452b[i5].k(h2);
            } else {
                h2 = this.f452b[i5].h(nVar.f2791g);
                i3 = this.f457g.f2791g;
            }
            int i6 = h2 - i3;
            if (i6 >= 0) {
                this.v[i4] = i6;
                i4++;
            }
        }
        Arrays.sort(this.v, 0, i4);
        for (int i7 = 0; i7 < i4; i7++) {
            int i8 = this.f457g.f2787c;
            if (!(i8 >= 0 && i8 < a0Var.b())) {
                return;
            }
            ((m.b) cVar).a(this.f457g.f2787c, this.v[i7]);
            n nVar2 = this.f457g;
            nVar2.f2787c += nVar2.f2788d;
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollExtent(RecyclerView.a0 a0Var) {
        return computeScrollExtent(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollOffset(RecyclerView.a0 a0Var) {
        return computeScrollOffset(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollRange(RecyclerView.a0 a0Var) {
        return computeScrollRange(a0Var);
    }

    public final int computeScrollExtent(RecyclerView.a0 a0Var) {
        if (getChildCount() == 0) {
            return 0;
        }
        return b.v.u.c.e(a0Var, this.f453c, e(!this.u), d(!this.u), this, this.u);
    }

    public final int computeScrollOffset(RecyclerView.a0 a0Var) {
        if (getChildCount() == 0) {
            return 0;
        }
        return b.v.u.c.f(a0Var, this.f453c, e(!this.u), d(!this.u), this, this.u, this.i);
    }

    public final int computeScrollRange(RecyclerView.a0 a0Var) {
        if (getChildCount() == 0) {
            return 0;
        }
        return b.v.u.c.g(a0Var, this.f453c, e(!this.u), d(!this.u), this, this.u);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.z.b
    public PointF computeScrollVectorForPosition(int i) {
        int a2 = a(i);
        PointF pointF = new PointF();
        if (a2 == 0) {
            return null;
        }
        if (this.f455e == 0) {
            pointF.x = a2;
            pointF.y = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        } else {
            pointF.x = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            pointF.y = a2;
        }
        return pointF;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollExtent(RecyclerView.a0 a0Var) {
        return computeScrollExtent(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollOffset(RecyclerView.a0 a0Var) {
        return computeScrollOffset(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollRange(RecyclerView.a0 a0Var) {
        return computeScrollRange(a0Var);
    }

    public View d(boolean z) {
        int k = this.f453c.k();
        int g2 = this.f453c.g();
        View view = null;
        for (int childCount = getChildCount() - 1; childCount >= 0; childCount--) {
            View childAt = getChildAt(childCount);
            int e2 = this.f453c.e(childAt);
            int b2 = this.f453c.b(childAt);
            if (b2 > k && e2 < g2) {
                if (b2 <= g2 || !z) {
                    return childAt;
                }
                if (view == null) {
                    view = childAt;
                }
            }
        }
        return view;
    }

    public View e(boolean z) {
        int k = this.f453c.k();
        int g2 = this.f453c.g();
        int childCount = getChildCount();
        View view = null;
        for (int i = 0; i < childCount; i++) {
            View childAt = getChildAt(i);
            int e2 = this.f453c.e(childAt);
            if (this.f453c.b(childAt) > k && e2 < g2) {
                if (e2 >= k || !z) {
                    return childAt;
                }
                if (view == null) {
                    view = childAt;
                }
            }
        }
        return view;
    }

    public final void f(RecyclerView.v vVar, RecyclerView.a0 a0Var, boolean z) {
        int g2;
        int j = j(Integer.MIN_VALUE);
        if (j != Integer.MIN_VALUE && (g2 = this.f453c.g() - j) > 0) {
            int i = g2 - (-scrollBy(-g2, vVar, a0Var));
            if (!z || i <= 0) {
                return;
            }
            this.f453c.p(i);
        }
    }

    public final void g(RecyclerView.v vVar, RecyclerView.a0 a0Var, boolean z) {
        int k;
        int k2 = k(Integer.MAX_VALUE);
        if (k2 != Integer.MAX_VALUE && (k = k2 - this.f453c.k()) > 0) {
            int scrollBy = k - scrollBy(k, vVar, a0Var);
            if (!z || scrollBy <= 0) {
                return;
            }
            this.f453c.p(-scrollBy);
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public RecyclerView.p generateDefaultLayoutParams() {
        if (this.f455e == 0) {
            return new c(-2, -1);
        }
        return new c(-1, -2);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public RecyclerView.p generateLayoutParams(Context context, AttributeSet attributeSet) {
        return new c(context, attributeSet);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int getColumnCountForAccessibility(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (this.f455e == 1) {
            return this.f451a;
        }
        return super.getColumnCountForAccessibility(vVar, a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int getRowCountForAccessibility(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (this.f455e == 0) {
            return this.f451a;
        }
        return super.getRowCountForAccessibility(vVar, a0Var);
    }

    public int h() {
        if (getChildCount() == 0) {
            return 0;
        }
        return getPosition(getChildAt(0));
    }

    public int i() {
        int childCount = getChildCount();
        if (childCount == 0) {
            return 0;
        }
        return getPosition(getChildAt(childCount - 1));
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean isAutoMeasureEnabled() {
        return this.n != 0;
    }

    public boolean isLayoutRTL() {
        return getLayoutDirection() == 1;
    }

    public final int j(int i) {
        int h2 = this.f452b[0].h(i);
        for (int i2 = 1; i2 < this.f451a; i2++) {
            int h3 = this.f452b[i2].h(i);
            if (h3 > h2) {
                h2 = h3;
            }
        }
        return h2;
    }

    public final int k(int i) {
        int k = this.f452b[0].k(i);
        for (int i2 = 1; i2 < this.f451a; i2++) {
            int k2 = this.f452b[i2].k(i);
            if (k2 < k) {
                k = k2;
            }
        }
        return k;
    }

    /* JADX WARN: Removed duplicated region for block: B:15:0x0025  */
    /* JADX WARN: Removed duplicated region for block: B:21:0x003c  */
    /* JADX WARN: Removed duplicated region for block: B:23:0x0043 A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:24:0x0044  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void l(int i, int i2, int i3) {
        int i4;
        int i5;
        int i6 = this.i ? i() : h();
        if (i3 != 8) {
            i4 = i + i2;
        } else if (i >= i2) {
            i4 = i + 1;
            i5 = i2;
            this.m.d(i5);
            if (i3 != 1) {
                this.m.e(i, i2);
            } else if (i3 == 2) {
                this.m.f(i, i2);
            } else if (i3 == 8) {
                this.m.f(i, 1);
                this.m.e(i2, 1);
            }
            if (i4 > i6) {
                return;
            }
            if (i5 <= (this.i ? h() : i())) {
                requestLayout();
                return;
            }
            return;
        } else {
            i4 = i2 + 1;
        }
        i5 = i;
        this.m.d(i5);
        if (i3 != 1) {
        }
        if (i4 > i6) {
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:45:0x00bc, code lost:
        if (r10 == r11) goto L58;
     */
    /* JADX WARN: Code restructure failed: missing block: B:50:0x00ce, code lost:
        if (r10 == r11) goto L58;
     */
    /* JADX WARN: Code restructure failed: missing block: B:51:0x00d0, code lost:
        r10 = true;
     */
    /* JADX WARN: Code restructure failed: missing block: B:52:0x00d2, code lost:
        r10 = false;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public View m() {
        int i;
        boolean z;
        boolean z2;
        int childCount = getChildCount() - 1;
        BitSet bitSet = new BitSet(this.f451a);
        bitSet.set(0, this.f451a, true);
        char c2 = (this.f455e == 1 && isLayoutRTL()) ? (char) 1 : (char) 65535;
        if (this.i) {
            i = -1;
        } else {
            i = childCount + 1;
            childCount = 0;
        }
        int i2 = childCount < i ? 1 : -1;
        while (childCount != i) {
            View childAt = getChildAt(childCount);
            c cVar = (c) childAt.getLayoutParams();
            if (bitSet.get(cVar.f467e.f485e)) {
                f fVar = cVar.f467e;
                if (this.i) {
                    int i3 = fVar.f483c;
                    if (i3 == Integer.MIN_VALUE) {
                        fVar.b();
                        i3 = fVar.f483c;
                    }
                    if (i3 < this.f453c.g()) {
                        ArrayList<View> arrayList = fVar.f481a;
                        Objects.requireNonNull(fVar.j(arrayList.get(arrayList.size() - 1)));
                        z2 = true;
                    }
                    z2 = false;
                } else {
                    int i4 = fVar.f482b;
                    if (i4 == Integer.MIN_VALUE) {
                        fVar.c();
                        i4 = fVar.f482b;
                    }
                    if (i4 > this.f453c.k()) {
                        Objects.requireNonNull(fVar.j(fVar.f481a.get(0)));
                        z2 = true;
                    }
                    z2 = false;
                }
                if (z2) {
                    return childAt;
                }
                bitSet.clear(cVar.f467e.f485e);
            }
            int i5 = childCount + i2;
            if (i5 != i) {
                View childAt2 = getChildAt(i5);
                if (this.i) {
                    int b2 = this.f453c.b(childAt);
                    int b3 = this.f453c.b(childAt2);
                    if (b2 < b3) {
                        return childAt;
                    }
                } else {
                    int e2 = this.f453c.e(childAt);
                    int e3 = this.f453c.e(childAt2);
                    if (e2 > e3) {
                        return childAt;
                    }
                }
                if (z) {
                    if ((cVar.f467e.f485e - ((c) childAt2.getLayoutParams()).f467e.f485e < 0) != (c2 < 0)) {
                        return childAt;
                    }
                } else {
                    continue;
                }
            }
            childCount += i2;
        }
        return null;
    }

    public final void n(View view, int i, int i2, boolean z) {
        boolean shouldMeasureChild;
        calculateItemDecorationsForChild(view, this.s);
        c cVar = (c) view.getLayoutParams();
        int i3 = ((ViewGroup.MarginLayoutParams) cVar).leftMargin;
        Rect rect = this.s;
        int y = y(i, i3 + rect.left, ((ViewGroup.MarginLayoutParams) cVar).rightMargin + rect.right);
        int i4 = ((ViewGroup.MarginLayoutParams) cVar).topMargin;
        Rect rect2 = this.s;
        int y2 = y(i2, i4 + rect2.top, ((ViewGroup.MarginLayoutParams) cVar).bottomMargin + rect2.bottom);
        if (z) {
            shouldMeasureChild = shouldReMeasureChild(view, y, y2, cVar);
        } else {
            shouldMeasureChild = shouldMeasureChild(view, y, y2, cVar);
        }
        if (shouldMeasureChild) {
            view.measure(y, y2);
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:243:0x040e, code lost:
        if (b() != false) goto L254;
     */
    /* JADX WARN: Removed duplicated region for block: B:112:0x01b9  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void o(RecyclerView.v vVar, RecyclerView.a0 a0Var, boolean z) {
        e eVar;
        int k;
        boolean z2;
        int i;
        int i2;
        int k2;
        int k3;
        b bVar = this.t;
        if ((this.q != null || this.k != -1) && a0Var.b() == 0) {
            removeAndRecycleAllViews(vVar);
            bVar.b();
            return;
        }
        boolean z3 = true;
        boolean z4 = (bVar.f464e && this.k == -1 && this.q == null) ? false : true;
        if (z4) {
            bVar.b();
            e eVar2 = this.q;
            if (eVar2 != null) {
                int i3 = eVar2.f476d;
                if (i3 > 0) {
                    if (i3 == this.f451a) {
                        for (int i4 = 0; i4 < this.f451a; i4++) {
                            this.f452b[i4].d();
                            e eVar3 = this.q;
                            int i5 = eVar3.f477e[i4];
                            if (i5 != Integer.MIN_VALUE) {
                                if (eVar3.j) {
                                    k3 = this.f453c.g();
                                } else {
                                    k3 = this.f453c.k();
                                }
                                i5 += k3;
                            }
                            f fVar = this.f452b[i4];
                            fVar.f482b = i5;
                            fVar.f483c = i5;
                        }
                    } else {
                        eVar2.f477e = null;
                        eVar2.f476d = 0;
                        eVar2.f478f = 0;
                        eVar2.f479g = null;
                        eVar2.f480h = null;
                        eVar2.f474b = eVar2.f475c;
                    }
                }
                e eVar4 = this.q;
                this.p = eVar4.k;
                boolean z5 = eVar4.i;
                assertNotInLayoutOrScroll(null);
                e eVar5 = this.q;
                if (eVar5 != null && eVar5.i != z5) {
                    eVar5.i = z5;
                }
                this.f458h = z5;
                requestLayout();
                resolveShouldLayoutReverse();
                e eVar6 = this.q;
                int i6 = eVar6.f474b;
                if (i6 != -1) {
                    this.k = i6;
                    bVar.f462c = eVar6.j;
                } else {
                    bVar.f462c = this.i;
                }
                if (eVar6.f478f > 1) {
                    d dVar = this.m;
                    dVar.f468a = eVar6.f479g;
                    dVar.f469b = eVar6.f480h;
                }
            } else {
                resolveShouldLayoutReverse();
                bVar.f462c = this.i;
            }
            if (!a0Var.f396g && (i2 = this.k) != -1) {
                if (i2 >= 0 && i2 < a0Var.b()) {
                    e eVar7 = this.q;
                    if (eVar7 != null && eVar7.f474b != -1 && eVar7.f476d >= 1) {
                        bVar.f461b = Integer.MIN_VALUE;
                        bVar.f460a = this.k;
                    } else {
                        View findViewByPosition = findViewByPosition(this.k);
                        if (findViewByPosition != null) {
                            bVar.f460a = this.i ? i() : h();
                            if (this.l != Integer.MIN_VALUE) {
                                if (bVar.f462c) {
                                    bVar.f461b = (this.f453c.g() - this.l) - this.f453c.b(findViewByPosition);
                                } else {
                                    bVar.f461b = (this.f453c.k() + this.l) - this.f453c.e(findViewByPosition);
                                }
                            } else if (this.f453c.c(findViewByPosition) > this.f453c.l()) {
                                if (bVar.f462c) {
                                    k2 = this.f453c.g();
                                } else {
                                    k2 = this.f453c.k();
                                }
                                bVar.f461b = k2;
                            } else {
                                int e2 = this.f453c.e(findViewByPosition) - this.f453c.k();
                                if (e2 < 0) {
                                    bVar.f461b = -e2;
                                } else {
                                    int g2 = this.f453c.g() - this.f453c.b(findViewByPosition);
                                    if (g2 < 0) {
                                        bVar.f461b = g2;
                                    } else {
                                        bVar.f461b = Integer.MIN_VALUE;
                                    }
                                }
                            }
                        } else {
                            int i7 = this.k;
                            bVar.f460a = i7;
                            int i8 = this.l;
                            if (i8 == Integer.MIN_VALUE) {
                                bVar.f462c = a(i7) == 1;
                                bVar.a();
                            } else if (bVar.f462c) {
                                bVar.f461b = StaggeredGridLayoutManager.this.f453c.g() - i8;
                            } else {
                                bVar.f461b = StaggeredGridLayoutManager.this.f453c.k() + i8;
                            }
                            bVar.f463d = true;
                        }
                    }
                    z2 = true;
                    if (!z2) {
                        if (this.o) {
                            int b2 = a0Var.b();
                            int childCount = getChildCount();
                            while (true) {
                                childCount--;
                                if (childCount < 0) {
                                    break;
                                }
                                i = getPosition(getChildAt(childCount));
                                if (i >= 0 && i < b2) {
                                    break;
                                }
                            }
                            i = 0;
                            bVar.f460a = i;
                            bVar.f461b = Integer.MIN_VALUE;
                        } else {
                            int b3 = a0Var.b();
                            int childCount2 = getChildCount();
                            for (int i9 = 0; i9 < childCount2; i9++) {
                                int position = getPosition(getChildAt(i9));
                                if (position >= 0 && position < b3) {
                                    i = position;
                                    break;
                                }
                            }
                            i = 0;
                            bVar.f460a = i;
                            bVar.f461b = Integer.MIN_VALUE;
                        }
                    }
                    bVar.f464e = true;
                } else {
                    this.k = -1;
                    this.l = Integer.MIN_VALUE;
                }
            }
            z2 = false;
            if (!z2) {
            }
            bVar.f464e = true;
        }
        if (this.q == null && this.k == -1 && (bVar.f462c != this.o || isLayoutRTL() != this.p)) {
            this.m.a();
            bVar.f463d = true;
        }
        if (getChildCount() > 0 && ((eVar = this.q) == null || eVar.f476d < 1)) {
            if (bVar.f463d) {
                for (int i10 = 0; i10 < this.f451a; i10++) {
                    this.f452b[i10].d();
                    int i11 = bVar.f461b;
                    if (i11 != Integer.MIN_VALUE) {
                        f fVar2 = this.f452b[i10];
                        fVar2.f482b = i11;
                        fVar2.f483c = i11;
                    }
                }
            } else if (!z4 && this.t.f465f != null) {
                for (int i12 = 0; i12 < this.f451a; i12++) {
                    f fVar3 = this.f452b[i12];
                    fVar3.d();
                    int i13 = this.t.f465f[i12];
                    fVar3.f482b = i13;
                    fVar3.f483c = i13;
                }
            } else {
                for (int i14 = 0; i14 < this.f451a; i14++) {
                    f fVar4 = this.f452b[i14];
                    boolean z6 = this.i;
                    int i15 = bVar.f461b;
                    if (z6) {
                        k = fVar4.h(Integer.MIN_VALUE);
                    } else {
                        k = fVar4.k(Integer.MIN_VALUE);
                    }
                    fVar4.d();
                    if (k != Integer.MIN_VALUE && ((!z6 || k >= StaggeredGridLayoutManager.this.f453c.g()) && (z6 || k <= StaggeredGridLayoutManager.this.f453c.k()))) {
                        if (i15 != Integer.MIN_VALUE) {
                            k += i15;
                        }
                        fVar4.f483c = k;
                        fVar4.f482b = k;
                    }
                }
                b bVar2 = this.t;
                f[] fVarArr = this.f452b;
                Objects.requireNonNull(bVar2);
                int length = fVarArr.length;
                int[] iArr = bVar2.f465f;
                if (iArr == null || iArr.length < length) {
                    bVar2.f465f = new int[StaggeredGridLayoutManager.this.f452b.length];
                }
                for (int i16 = 0; i16 < length; i16++) {
                    bVar2.f465f[i16] = fVarArr[i16].k(Integer.MIN_VALUE);
                }
            }
        }
        detachAndScrapAttachedViews(vVar);
        this.f457g.f2785a = false;
        int l = this.f454d.l();
        this.f456f = l / this.f451a;
        this.r = View.MeasureSpec.makeMeasureSpec(l, this.f454d.i());
        w(bVar.f460a, a0Var);
        if (bVar.f462c) {
            u(-1);
            c(vVar, this.f457g, a0Var);
            u(1);
            n nVar = this.f457g;
            nVar.f2787c = bVar.f460a + nVar.f2788d;
            c(vVar, nVar, a0Var);
        } else {
            u(1);
            c(vVar, this.f457g, a0Var);
            u(-1);
            n nVar2 = this.f457g;
            nVar2.f2787c = bVar.f460a + nVar2.f2788d;
            c(vVar, nVar2, a0Var);
        }
        if (this.f454d.i() != 1073741824) {
            float f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            int childCount3 = getChildCount();
            for (int i17 = 0; i17 < childCount3; i17++) {
                View childAt = getChildAt(i17);
                float c2 = this.f454d.c(childAt);
                if (c2 >= f2) {
                    Objects.requireNonNull((c) childAt.getLayoutParams());
                    f2 = Math.max(f2, c2);
                }
            }
            int i18 = this.f456f;
            int round = Math.round(f2 * this.f451a);
            if (this.f454d.i() == Integer.MIN_VALUE) {
                round = Math.min(round, this.f454d.l());
            }
            this.f456f = round / this.f451a;
            this.r = View.MeasureSpec.makeMeasureSpec(round, this.f454d.i());
            if (this.f456f != i18) {
                for (int i19 = 0; i19 < childCount3; i19++) {
                    View childAt2 = getChildAt(i19);
                    c cVar = (c) childAt2.getLayoutParams();
                    Objects.requireNonNull(cVar);
                    if (isLayoutRTL() && this.f455e == 1) {
                        int i20 = this.f451a;
                        int i21 = cVar.f467e.f485e;
                        childAt2.offsetLeftAndRight(((-((i20 - 1) - i21)) * this.f456f) - ((-((i20 - 1) - i21)) * i18));
                    } else {
                        int i22 = cVar.f467e.f485e;
                        int i23 = this.f456f * i22;
                        int i24 = i22 * i18;
                        if (this.f455e == 1) {
                            childAt2.offsetLeftAndRight(i23 - i24);
                        } else {
                            childAt2.offsetTopAndBottom(i23 - i24);
                        }
                    }
                }
            }
        }
        if (getChildCount() > 0) {
            if (this.i) {
                f(vVar, a0Var, true);
                g(vVar, a0Var, false);
            } else {
                g(vVar, a0Var, true);
                f(vVar, a0Var, false);
            }
        }
        if (z && !a0Var.f396g) {
            if ((this.n == 0 || getChildCount() <= 0 || m() == null) ? false : true) {
                removeCallbacks(this.w);
            }
        }
        z3 = false;
        if (a0Var.f396g) {
            this.t.b();
        }
        this.o = bVar.f462c;
        this.p = isLayoutRTL();
        if (z3) {
            this.t.b();
            o(vVar, a0Var, false);
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void offsetChildrenHorizontal(int i) {
        super.offsetChildrenHorizontal(i);
        for (int i2 = 0; i2 < this.f451a; i2++) {
            f fVar = this.f452b[i2];
            int i3 = fVar.f482b;
            if (i3 != Integer.MIN_VALUE) {
                fVar.f482b = i3 + i;
            }
            int i4 = fVar.f483c;
            if (i4 != Integer.MIN_VALUE) {
                fVar.f483c = i4 + i;
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void offsetChildrenVertical(int i) {
        super.offsetChildrenVertical(i);
        for (int i2 = 0; i2 < this.f451a; i2++) {
            f fVar = this.f452b[i2];
            int i3 = fVar.f482b;
            if (i3 != Integer.MIN_VALUE) {
                fVar.f482b = i3 + i;
            }
            int i4 = fVar.f483c;
            if (i4 != Integer.MIN_VALUE) {
                fVar.f483c = i4 + i;
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onDetachedFromWindow(RecyclerView recyclerView, RecyclerView.v vVar) {
        super.onDetachedFromWindow(recyclerView, vVar);
        removeCallbacks(this.w);
        for (int i = 0; i < this.f451a; i++) {
            this.f452b[i].d();
        }
        recyclerView.requestLayout();
    }

    /* JADX WARN: Code restructure failed: missing block: B:29:0x003b, code lost:
        if (r8.f455e == 1) goto L111;
     */
    /* JADX WARN: Code restructure failed: missing block: B:32:0x0041, code lost:
        if (r8.f455e == 0) goto L111;
     */
    /* JADX WARN: Code restructure failed: missing block: B:38:0x004d, code lost:
        if (isLayoutRTL() == false) goto L22;
     */
    /* JADX WARN: Code restructure failed: missing block: B:44:0x0059, code lost:
        if (isLayoutRTL() == false) goto L111;
     */
    @Override // androidx.recyclerview.widget.RecyclerView.o
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public View onFocusSearchFailed(View view, int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        View findContainingItemView;
        int i2;
        int h2;
        int f2;
        int f3;
        int f4;
        if (getChildCount() == 0 || (findContainingItemView = findContainingItemView(view)) == null) {
            return null;
        }
        resolveShouldLayoutReverse();
        if (i == 1) {
            if (this.f455e != 1) {
            }
            i2 = -1;
        } else if (i == 2) {
            if (this.f455e != 1) {
            }
            i2 = 1;
        } else if (i != 17) {
            if (i != 33) {
                if (i == 66) {
                }
            }
            i2 = Integer.MIN_VALUE;
        }
        if (i2 == Integer.MIN_VALUE) {
            return null;
        }
        c cVar = (c) findContainingItemView.getLayoutParams();
        Objects.requireNonNull(cVar);
        f fVar = cVar.f467e;
        if (i2 == 1) {
            h2 = i();
        } else {
            h2 = h();
        }
        w(h2, a0Var);
        u(i2);
        n nVar = this.f457g;
        nVar.f2787c = nVar.f2788d + h2;
        nVar.f2786b = (int) (this.f453c.l() * 0.33333334f);
        n nVar2 = this.f457g;
        nVar2.f2792h = true;
        nVar2.f2785a = false;
        c(vVar, nVar2, a0Var);
        this.o = this.i;
        View i3 = fVar.i(h2, i2);
        if (i3 == null || i3 == findContainingItemView) {
            if (p(i2)) {
                for (int i4 = this.f451a - 1; i4 >= 0; i4--) {
                    View i5 = this.f452b[i4].i(h2, i2);
                    if (i5 != null && i5 != findContainingItemView) {
                        return i5;
                    }
                }
            } else {
                for (int i6 = 0; i6 < this.f451a; i6++) {
                    View i7 = this.f452b[i6].i(h2, i2);
                    if (i7 != null && i7 != findContainingItemView) {
                        return i7;
                    }
                }
            }
            boolean z = (this.f458h ^ true) == (i2 == -1);
            if (z) {
                f2 = fVar.e();
            } else {
                f2 = fVar.f();
            }
            View findViewByPosition = findViewByPosition(f2);
            if (findViewByPosition == null || findViewByPosition == findContainingItemView) {
                if (p(i2)) {
                    for (int i8 = this.f451a - 1; i8 >= 0; i8--) {
                        if (i8 != fVar.f485e) {
                            if (z) {
                                f4 = this.f452b[i8].e();
                            } else {
                                f4 = this.f452b[i8].f();
                            }
                            View findViewByPosition2 = findViewByPosition(f4);
                            if (findViewByPosition2 != null && findViewByPosition2 != findContainingItemView) {
                                return findViewByPosition2;
                            }
                        }
                    }
                } else {
                    for (int i9 = 0; i9 < this.f451a; i9++) {
                        if (z) {
                            f3 = this.f452b[i9].e();
                        } else {
                            f3 = this.f452b[i9].f();
                        }
                        View findViewByPosition3 = findViewByPosition(f3);
                        if (findViewByPosition3 != null && findViewByPosition3 != findContainingItemView) {
                            return findViewByPosition3;
                        }
                    }
                }
                return null;
            }
            return findViewByPosition;
        }
        return i3;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onInitializeAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        super.onInitializeAccessibilityEvent(accessibilityEvent);
        if (getChildCount() > 0) {
            View e2 = e(false);
            View d2 = d(false);
            if (e2 == null || d2 == null) {
                return;
            }
            int position = getPosition(e2);
            int position2 = getPosition(d2);
            if (position < position2) {
                accessibilityEvent.setFromIndex(position);
                accessibilityEvent.setToIndex(position2);
                return;
            }
            accessibilityEvent.setFromIndex(position2);
            accessibilityEvent.setToIndex(position);
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onInitializeAccessibilityNodeInfoForItem(RecyclerView.v vVar, RecyclerView.a0 a0Var, View view, b.j.j.x.b bVar) {
        ViewGroup.LayoutParams layoutParams = view.getLayoutParams();
        if (!(layoutParams instanceof c)) {
            super.onInitializeAccessibilityNodeInfoForItem(view, bVar);
            return;
        }
        c cVar = (c) layoutParams;
        if (this.f455e == 0) {
            f fVar = cVar.f467e;
            bVar.n(b.c.a(fVar != null ? fVar.f485e : -1, 1, -1, -1, false, false));
            return;
        }
        f fVar2 = cVar.f467e;
        bVar.n(b.c.a(-1, -1, fVar2 != null ? fVar2.f485e : -1, 1, false, false));
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsAdded(RecyclerView recyclerView, int i, int i2) {
        l(i, i2, 1);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsChanged(RecyclerView recyclerView) {
        this.m.a();
        requestLayout();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsMoved(RecyclerView recyclerView, int i, int i2, int i3) {
        l(i, i2, 8);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsRemoved(RecyclerView recyclerView, int i, int i2) {
        l(i, i2, 2);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsUpdated(RecyclerView recyclerView, int i, int i2, Object obj) {
        l(i, i2, 4);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onLayoutChildren(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        o(vVar, a0Var, true);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onLayoutCompleted(RecyclerView.a0 a0Var) {
        super.onLayoutCompleted(a0Var);
        this.k = -1;
        this.l = Integer.MIN_VALUE;
        this.q = null;
        this.t.b();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (parcelable instanceof e) {
            this.q = (e) parcelable;
            requestLayout();
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public Parcelable onSaveInstanceState() {
        int k;
        int k2;
        int[] iArr;
        e eVar = this.q;
        if (eVar != null) {
            return new e(eVar);
        }
        e eVar2 = new e();
        eVar2.i = this.f458h;
        eVar2.j = this.o;
        eVar2.k = this.p;
        d dVar = this.m;
        if (dVar != null && (iArr = dVar.f468a) != null) {
            eVar2.f479g = iArr;
            eVar2.f478f = iArr.length;
            eVar2.f480h = dVar.f469b;
        } else {
            eVar2.f478f = 0;
        }
        if (getChildCount() > 0) {
            eVar2.f474b = this.o ? i() : h();
            View d2 = this.i ? d(true) : e(true);
            eVar2.f475c = d2 != null ? getPosition(d2) : -1;
            int i = this.f451a;
            eVar2.f476d = i;
            eVar2.f477e = new int[i];
            for (int i2 = 0; i2 < this.f451a; i2++) {
                if (this.o) {
                    k = this.f452b[i2].h(Integer.MIN_VALUE);
                    if (k != Integer.MIN_VALUE) {
                        k2 = this.f453c.g();
                        k -= k2;
                        eVar2.f477e[i2] = k;
                    } else {
                        eVar2.f477e[i2] = k;
                    }
                } else {
                    k = this.f452b[i2].k(Integer.MIN_VALUE);
                    if (k != Integer.MIN_VALUE) {
                        k2 = this.f453c.k();
                        k -= k2;
                        eVar2.f477e[i2] = k;
                    } else {
                        eVar2.f477e[i2] = k;
                    }
                }
            }
        } else {
            eVar2.f474b = -1;
            eVar2.f475c = -1;
            eVar2.f476d = 0;
        }
        return eVar2;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onScrollStateChanged(int i) {
        if (i == 0) {
            b();
        }
    }

    public final boolean p(int i) {
        if (this.f455e == 0) {
            return (i == -1) != this.i;
        }
        return ((i == -1) == this.i) == isLayoutRTL();
    }

    public void q(int i, RecyclerView.a0 a0Var) {
        int i2;
        int h2;
        if (i > 0) {
            h2 = i();
            i2 = 1;
        } else {
            i2 = -1;
            h2 = h();
        }
        this.f457g.f2785a = true;
        w(h2, a0Var);
        u(i2);
        n nVar = this.f457g;
        nVar.f2787c = h2 + nVar.f2788d;
        nVar.f2786b = Math.abs(i);
    }

    public final void r(RecyclerView.v vVar, n nVar) {
        int min;
        int min2;
        if (!nVar.f2785a || nVar.i) {
            return;
        }
        if (nVar.f2786b == 0) {
            if (nVar.f2789e == -1) {
                s(vVar, nVar.f2791g);
                return;
            } else {
                t(vVar, nVar.f2790f);
                return;
            }
        }
        int i = 1;
        if (nVar.f2789e == -1) {
            int i2 = nVar.f2790f;
            int k = this.f452b[0].k(i2);
            while (i < this.f451a) {
                int k2 = this.f452b[i].k(i2);
                if (k2 > k) {
                    k = k2;
                }
                i++;
            }
            int i3 = i2 - k;
            if (i3 < 0) {
                min2 = nVar.f2791g;
            } else {
                min2 = nVar.f2791g - Math.min(i3, nVar.f2786b);
            }
            s(vVar, min2);
            return;
        }
        int i4 = nVar.f2791g;
        int h2 = this.f452b[0].h(i4);
        while (i < this.f451a) {
            int h3 = this.f452b[i].h(i4);
            if (h3 < h2) {
                h2 = h3;
            }
            i++;
        }
        int i5 = h2 - nVar.f2791g;
        if (i5 < 0) {
            min = nVar.f2790f;
        } else {
            min = Math.min(i5, nVar.f2786b) + nVar.f2790f;
        }
        t(vVar, min);
    }

    public final void resolveShouldLayoutReverse() {
        if (this.f455e != 1 && isLayoutRTL()) {
            this.i = !this.f458h;
        } else {
            this.i = this.f458h;
        }
    }

    public final void s(RecyclerView.v vVar, int i) {
        for (int childCount = getChildCount() - 1; childCount >= 0; childCount--) {
            View childAt = getChildAt(childCount);
            if (this.f453c.e(childAt) < i || this.f453c.o(childAt) < i) {
                return;
            }
            c cVar = (c) childAt.getLayoutParams();
            Objects.requireNonNull(cVar);
            if (cVar.f467e.f481a.size() == 1) {
                return;
            }
            cVar.f467e.l();
            removeAndRecycleView(childAt, vVar);
        }
    }

    public int scrollBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (getChildCount() == 0 || i == 0) {
            return 0;
        }
        q(i, a0Var);
        int c2 = c(vVar, this.f457g, a0Var);
        if (this.f457g.f2786b >= c2) {
            i = i < 0 ? -c2 : c2;
        }
        this.f453c.p(-i);
        this.o = this.i;
        n nVar = this.f457g;
        nVar.f2786b = 0;
        r(vVar, nVar);
        return i;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int scrollHorizontallyBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        return scrollBy(i, vVar, a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void scrollToPosition(int i) {
        e eVar = this.q;
        if (eVar != null && eVar.f474b != i) {
            eVar.f477e = null;
            eVar.f476d = 0;
            eVar.f474b = -1;
            eVar.f475c = -1;
        }
        this.k = i;
        this.l = Integer.MIN_VALUE;
        requestLayout();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int scrollVerticallyBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        return scrollBy(i, vVar, a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void setMeasuredDimension(Rect rect, int i, int i2) {
        int chooseSize;
        int chooseSize2;
        int paddingRight = getPaddingRight() + getPaddingLeft();
        int paddingBottom = getPaddingBottom() + getPaddingTop();
        if (this.f455e == 1) {
            chooseSize2 = RecyclerView.o.chooseSize(i2, rect.height() + paddingBottom, getMinimumHeight());
            chooseSize = RecyclerView.o.chooseSize(i, (this.f456f * this.f451a) + paddingRight, getMinimumWidth());
        } else {
            chooseSize = RecyclerView.o.chooseSize(i, rect.width() + paddingRight, getMinimumWidth());
            chooseSize2 = RecyclerView.o.chooseSize(i2, (this.f456f * this.f451a) + paddingBottom, getMinimumHeight());
        }
        setMeasuredDimension(chooseSize, chooseSize2);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void smoothScrollToPosition(RecyclerView recyclerView, RecyclerView.a0 a0Var, int i) {
        o oVar = new o(recyclerView.getContext());
        oVar.setTargetPosition(i);
        startSmoothScroll(oVar);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean supportsPredictiveItemAnimations() {
        return this.q == null;
    }

    public final void t(RecyclerView.v vVar, int i) {
        while (getChildCount() > 0) {
            View childAt = getChildAt(0);
            if (this.f453c.b(childAt) > i || this.f453c.n(childAt) > i) {
                return;
            }
            c cVar = (c) childAt.getLayoutParams();
            Objects.requireNonNull(cVar);
            if (cVar.f467e.f481a.size() == 1) {
                return;
            }
            cVar.f467e.m();
            removeAndRecycleView(childAt, vVar);
        }
    }

    public final void u(int i) {
        n nVar = this.f457g;
        nVar.f2789e = i;
        nVar.f2788d = this.i != (i == -1) ? -1 : 1;
    }

    public final void v(int i, int i2) {
        for (int i3 = 0; i3 < this.f451a; i3++) {
            if (!this.f452b[i3].f481a.isEmpty()) {
                x(this.f452b[i3], i, i2);
            }
        }
    }

    public final void w(int i, RecyclerView.a0 a0Var) {
        int i2;
        int i3;
        int i4;
        n nVar = this.f457g;
        boolean z = false;
        nVar.f2786b = 0;
        nVar.f2787c = i;
        if (!isSmoothScrolling() || (i4 = a0Var.f390a) == -1) {
            i2 = 0;
            i3 = 0;
        } else {
            if (this.i == (i4 < i)) {
                i2 = this.f453c.l();
                i3 = 0;
            } else {
                i3 = this.f453c.l();
                i2 = 0;
            }
        }
        if (getClipToPadding()) {
            this.f457g.f2790f = this.f453c.k() - i3;
            this.f457g.f2791g = this.f453c.g() + i2;
        } else {
            this.f457g.f2791g = this.f453c.f() + i2;
            this.f457g.f2790f = -i3;
        }
        n nVar2 = this.f457g;
        nVar2.f2792h = false;
        nVar2.f2785a = true;
        if (this.f453c.i() == 0 && this.f453c.f() == 0) {
            z = true;
        }
        nVar2.i = z;
    }

    public final void x(f fVar, int i, int i2) {
        int i3 = fVar.f484d;
        if (i == -1) {
            int i4 = fVar.f482b;
            if (i4 == Integer.MIN_VALUE) {
                fVar.c();
                i4 = fVar.f482b;
            }
            if (i4 + i3 <= i2) {
                this.j.set(fVar.f485e, false);
                return;
            }
            return;
        }
        int i5 = fVar.f483c;
        if (i5 == Integer.MIN_VALUE) {
            fVar.b();
            i5 = fVar.f483c;
        }
        if (i5 - i3 >= i2) {
            this.j.set(fVar.f485e, false);
        }
    }

    public final int y(int i, int i2, int i3) {
        if (i2 == 0 && i3 == 0) {
            return i;
        }
        int mode = View.MeasureSpec.getMode(i);
        return (mode == Integer.MIN_VALUE || mode == 1073741824) ? View.MeasureSpec.makeMeasureSpec(Math.max(0, (View.MeasureSpec.getSize(i) - i2) - i3), mode) : i;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public RecyclerView.p generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        if (layoutParams instanceof ViewGroup.MarginLayoutParams) {
            return new c((ViewGroup.MarginLayoutParams) layoutParams);
        }
        return new c(layoutParams);
    }

    /* loaded from: classes.dex */
    public static class d {

        /* renamed from: a  reason: collision with root package name */
        public int[] f468a;

        /* renamed from: b  reason: collision with root package name */
        public List<a> f469b;

        public void a() {
            int[] iArr = this.f468a;
            if (iArr != null) {
                Arrays.fill(iArr, -1);
            }
            this.f469b = null;
        }

        public void b(int i) {
            int[] iArr = this.f468a;
            if (iArr == null) {
                int[] iArr2 = new int[Math.max(i, 10) + 1];
                this.f468a = iArr2;
                Arrays.fill(iArr2, -1);
            } else if (i >= iArr.length) {
                int length = iArr.length;
                while (length <= i) {
                    length *= 2;
                }
                int[] iArr3 = new int[length];
                this.f468a = iArr3;
                System.arraycopy(iArr, 0, iArr3, 0, iArr.length);
                int[] iArr4 = this.f468a;
                Arrays.fill(iArr4, iArr.length, iArr4.length, -1);
            }
        }

        public a c(int i) {
            List<a> list = this.f469b;
            if (list == null) {
                return null;
            }
            for (int size = list.size() - 1; size >= 0; size--) {
                a aVar = this.f469b.get(size);
                if (aVar.f470b == i) {
                    return aVar;
                }
            }
            return null;
        }

        /* JADX WARN: Removed duplicated region for block: B:24:0x0048  */
        /* JADX WARN: Removed duplicated region for block: B:26:0x0052  */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public int d(int i) {
            int i2;
            int[] iArr = this.f468a;
            if (iArr == null || i >= iArr.length) {
                return -1;
            }
            if (this.f469b != null) {
                a c2 = c(i);
                if (c2 != null) {
                    this.f469b.remove(c2);
                }
                int size = this.f469b.size();
                int i3 = 0;
                while (true) {
                    if (i3 >= size) {
                        i3 = -1;
                        break;
                    } else if (this.f469b.get(i3).f470b >= i) {
                        break;
                    } else {
                        i3++;
                    }
                }
                if (i3 != -1) {
                    this.f469b.remove(i3);
                    i2 = this.f469b.get(i3).f470b;
                    if (i2 != -1) {
                        int[] iArr2 = this.f468a;
                        Arrays.fill(iArr2, i, iArr2.length, -1);
                        return this.f468a.length;
                    }
                    int i4 = i2 + 1;
                    Arrays.fill(this.f468a, i, i4, -1);
                    return i4;
                }
            }
            i2 = -1;
            if (i2 != -1) {
            }
        }

        public void e(int i, int i2) {
            int[] iArr = this.f468a;
            if (iArr == null || i >= iArr.length) {
                return;
            }
            int i3 = i + i2;
            b(i3);
            int[] iArr2 = this.f468a;
            System.arraycopy(iArr2, i, iArr2, i3, (iArr2.length - i) - i2);
            Arrays.fill(this.f468a, i, i3, -1);
            List<a> list = this.f469b;
            if (list == null) {
                return;
            }
            for (int size = list.size() - 1; size >= 0; size--) {
                a aVar = this.f469b.get(size);
                int i4 = aVar.f470b;
                if (i4 >= i) {
                    aVar.f470b = i4 + i2;
                }
            }
        }

        public void f(int i, int i2) {
            int[] iArr = this.f468a;
            if (iArr == null || i >= iArr.length) {
                return;
            }
            int i3 = i + i2;
            b(i3);
            int[] iArr2 = this.f468a;
            System.arraycopy(iArr2, i3, iArr2, i, (iArr2.length - i) - i2);
            int[] iArr3 = this.f468a;
            Arrays.fill(iArr3, iArr3.length - i2, iArr3.length, -1);
            List<a> list = this.f469b;
            if (list == null) {
                return;
            }
            for (int size = list.size() - 1; size >= 0; size--) {
                a aVar = this.f469b.get(size);
                int i4 = aVar.f470b;
                if (i4 >= i) {
                    if (i4 < i3) {
                        this.f469b.remove(size);
                    } else {
                        aVar.f470b = i4 - i2;
                    }
                }
            }
        }

        @SuppressLint({"BanParcelableUsage"})
        /* loaded from: classes.dex */
        public static class a implements Parcelable {
            public static final Parcelable.Creator<a> CREATOR = new C0004a();

            /* renamed from: b  reason: collision with root package name */
            public int f470b;

            /* renamed from: c  reason: collision with root package name */
            public int f471c;

            /* renamed from: d  reason: collision with root package name */
            public int[] f472d;

            /* renamed from: e  reason: collision with root package name */
            public boolean f473e;

            /* renamed from: androidx.recyclerview.widget.StaggeredGridLayoutManager$d$a$a  reason: collision with other inner class name */
            /* loaded from: classes.dex */
            public static class C0004a implements Parcelable.Creator<a> {
                /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
                @Override // android.os.Parcelable.Creator
                public a createFromParcel(Parcel parcel) {
                    return new a(parcel);
                }

                /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
                @Override // android.os.Parcelable.Creator
                public a[] newArray(int i) {
                    return new a[i];
                }
            }

            public a(Parcel parcel) {
                this.f470b = parcel.readInt();
                this.f471c = parcel.readInt();
                this.f473e = parcel.readInt() == 1;
                int readInt = parcel.readInt();
                if (readInt > 0) {
                    int[] iArr = new int[readInt];
                    this.f472d = iArr;
                    parcel.readIntArray(iArr);
                }
            }

            @Override // android.os.Parcelable
            public int describeContents() {
                return 0;
            }

            public String toString() {
                StringBuilder x = c.b.a.a.a.x("FullSpanItem{mPosition=");
                x.append(this.f470b);
                x.append(", mGapDir=");
                x.append(this.f471c);
                x.append(", mHasUnwantedGapAfter=");
                x.append(this.f473e);
                x.append(", mGapPerSpan=");
                x.append(Arrays.toString(this.f472d));
                x.append('}');
                return x.toString();
            }

            @Override // android.os.Parcelable
            public void writeToParcel(Parcel parcel, int i) {
                parcel.writeInt(this.f470b);
                parcel.writeInt(this.f471c);
                parcel.writeInt(this.f473e ? 1 : 0);
                int[] iArr = this.f472d;
                if (iArr != null && iArr.length > 0) {
                    parcel.writeInt(iArr.length);
                    parcel.writeIntArray(this.f472d);
                    return;
                }
                parcel.writeInt(0);
            }

            public a() {
            }
        }
    }
}