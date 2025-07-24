package b.w.b;

import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* compiled from: ChildHelper.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public final InterfaceC0054b f2712a;

    /* renamed from: b  reason: collision with root package name */
    public final a f2713b = new a();

    /* renamed from: c  reason: collision with root package name */
    public final List<View> f2714c = new ArrayList();

    /* compiled from: ChildHelper.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public long f2715a = 0;

        /* renamed from: b  reason: collision with root package name */
        public a f2716b;

        public void a(int i) {
            if (i >= 64) {
                a aVar = this.f2716b;
                if (aVar != null) {
                    aVar.a(i - 64);
                    return;
                }
                return;
            }
            this.f2715a &= ~(1 << i);
        }

        public int b(int i) {
            a aVar = this.f2716b;
            if (aVar == null) {
                if (i >= 64) {
                    return Long.bitCount(this.f2715a);
                }
                return Long.bitCount(this.f2715a & ((1 << i) - 1));
            } else if (i < 64) {
                return Long.bitCount(this.f2715a & ((1 << i) - 1));
            } else {
                return Long.bitCount(this.f2715a) + aVar.b(i - 64);
            }
        }

        public final void c() {
            if (this.f2716b == null) {
                this.f2716b = new a();
            }
        }

        public boolean d(int i) {
            if (i < 64) {
                return (this.f2715a & (1 << i)) != 0;
            }
            c();
            return this.f2716b.d(i - 64);
        }

        public void e(int i, boolean z) {
            if (i >= 64) {
                c();
                this.f2716b.e(i - 64, z);
                return;
            }
            long j = this.f2715a;
            boolean z2 = (Long.MIN_VALUE & j) != 0;
            long j2 = (1 << i) - 1;
            this.f2715a = ((j & (~j2)) << 1) | (j & j2);
            if (z) {
                h(i);
            } else {
                a(i);
            }
            if (z2 || this.f2716b != null) {
                c();
                this.f2716b.e(0, z2);
            }
        }

        public boolean f(int i) {
            if (i >= 64) {
                c();
                return this.f2716b.f(i - 64);
            }
            long j = 1 << i;
            long j2 = this.f2715a;
            boolean z = (j2 & j) != 0;
            long j3 = j2 & (~j);
            this.f2715a = j3;
            long j4 = j - 1;
            this.f2715a = (j3 & j4) | Long.rotateRight((~j4) & j3, 1);
            a aVar = this.f2716b;
            if (aVar != null) {
                if (aVar.d(0)) {
                    h(63);
                }
                this.f2716b.f(0);
            }
            return z;
        }

        public void g() {
            this.f2715a = 0L;
            a aVar = this.f2716b;
            if (aVar != null) {
                aVar.g();
            }
        }

        public void h(int i) {
            if (i >= 64) {
                c();
                this.f2716b.h(i - 64);
                return;
            }
            this.f2715a |= 1 << i;
        }

        public String toString() {
            if (this.f2716b == null) {
                return Long.toBinaryString(this.f2715a);
            }
            return this.f2716b.toString() + "xx" + Long.toBinaryString(this.f2715a);
        }
    }

    /* compiled from: ChildHelper.java */
    /* renamed from: b.w.b.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public interface InterfaceC0054b {
    }

    public b(InterfaceC0054b interfaceC0054b) {
        this.f2712a = interfaceC0054b;
    }

    public void a(View view, int i, boolean z) {
        int f2;
        if (i < 0) {
            f2 = ((RecyclerView.e) this.f2712a).b();
        } else {
            f2 = f(i);
        }
        this.f2713b.e(f2, z);
        if (z) {
            i(view);
        }
        RecyclerView.e eVar = (RecyclerView.e) this.f2712a;
        RecyclerView.this.addView(view, f2);
        RecyclerView.this.dispatchChildAttached(view);
    }

    public void b(View view, int i, ViewGroup.LayoutParams layoutParams, boolean z) {
        int f2;
        if (i < 0) {
            f2 = ((RecyclerView.e) this.f2712a).b();
        } else {
            f2 = f(i);
        }
        this.f2713b.e(f2, z);
        if (z) {
            i(view);
        }
        RecyclerView.e eVar = (RecyclerView.e) this.f2712a;
        Objects.requireNonNull(eVar);
        RecyclerView.d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
        if (childViewHolderInt != null) {
            if (!childViewHolderInt.isTmpDetached() && !childViewHolderInt.shouldIgnore()) {
                StringBuilder sb = new StringBuilder();
                sb.append("Called attach on a child which is not detached: ");
                sb.append(childViewHolderInt);
                throw new IllegalArgumentException(c.b.a.a.a.i(RecyclerView.this, sb));
            }
            childViewHolderInt.clearTmpDetachFlag();
        }
        RecyclerView.this.attachViewToParent(view, f2, layoutParams);
    }

    public void c(int i) {
        RecyclerView.d0 childViewHolderInt;
        int f2 = f(i);
        this.f2713b.f(f2);
        RecyclerView.e eVar = (RecyclerView.e) this.f2712a;
        View childAt = RecyclerView.this.getChildAt(f2);
        if (childAt != null && (childViewHolderInt = RecyclerView.getChildViewHolderInt(childAt)) != null) {
            if (childViewHolderInt.isTmpDetached() && !childViewHolderInt.shouldIgnore()) {
                StringBuilder sb = new StringBuilder();
                sb.append("called detach on an already detached child ");
                sb.append(childViewHolderInt);
                throw new IllegalArgumentException(c.b.a.a.a.i(RecyclerView.this, sb));
            }
            childViewHolderInt.addFlags(256);
        }
        RecyclerView.this.detachViewFromParent(f2);
    }

    public View d(int i) {
        return ((RecyclerView.e) this.f2712a).a(f(i));
    }

    public int e() {
        return ((RecyclerView.e) this.f2712a).b() - this.f2714c.size();
    }

    public final int f(int i) {
        if (i < 0) {
            return -1;
        }
        int b2 = ((RecyclerView.e) this.f2712a).b();
        int i2 = i;
        while (i2 < b2) {
            int b3 = i - (i2 - this.f2713b.b(i2));
            if (b3 == 0) {
                while (this.f2713b.d(i2)) {
                    i2++;
                }
                return i2;
            }
            i2 += b3;
        }
        return -1;
    }

    public View g(int i) {
        return RecyclerView.this.getChildAt(i);
    }

    public int h() {
        return ((RecyclerView.e) this.f2712a).b();
    }

    public final void i(View view) {
        this.f2714c.add(view);
        RecyclerView.e eVar = (RecyclerView.e) this.f2712a;
        Objects.requireNonNull(eVar);
        RecyclerView.d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
        if (childViewHolderInt != null) {
            childViewHolderInt.onEnteredHiddenState(RecyclerView.this);
        }
    }

    public int j(View view) {
        int indexOfChild = RecyclerView.this.indexOfChild(view);
        if (indexOfChild == -1 || this.f2713b.d(indexOfChild)) {
            return -1;
        }
        return indexOfChild - this.f2713b.b(indexOfChild);
    }

    public boolean k(View view) {
        return this.f2714c.contains(view);
    }

    public void l(int i) {
        int f2 = f(i);
        View a2 = ((RecyclerView.e) this.f2712a).a(f2);
        if (a2 == null) {
            return;
        }
        if (this.f2713b.f(f2)) {
            m(a2);
        }
        ((RecyclerView.e) this.f2712a).c(f2);
    }

    public final boolean m(View view) {
        if (this.f2714c.remove(view)) {
            RecyclerView.e eVar = (RecyclerView.e) this.f2712a;
            Objects.requireNonNull(eVar);
            RecyclerView.d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            if (childViewHolderInt != null) {
                childViewHolderInt.onLeftHiddenState(RecyclerView.this);
                return true;
            }
            return true;
        }
        return false;
    }

    public String toString() {
        return this.f2713b.toString() + ", hidden list:" + this.f2714c.size();
    }
}