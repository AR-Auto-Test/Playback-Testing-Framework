package c.e.b.ff;

import android.util.SparseIntArray;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.databinding.ViewDataBinding;
import com.ibosoninnov.unitear.R;

/* compiled from: ItemHistoryBindingImpl.java */
/* loaded from: classes2.dex */
public class h extends g {
    public static final SparseIntArray u;
    public final TextView v;
    public final View w;
    public long x;

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        u = sparseIntArray;
        sparseIntArray.put(R.id.imIcon, 3);
    }

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public h(b.m.e eVar, View view) {
        super(eVar, view, 1, (ImageView) r0[3], (LinearLayout) r0[0]);
        Object[] h2 = ViewDataBinding.h(eVar, view, 4, u);
        this.x = -1L;
        this.s.setTag(null);
        TextView textView = (TextView) h2[1];
        this.v = textView;
        textView.setTag(null);
        View view2 = (View) h2[2];
        this.w = view2;
        view2.setTag(null);
        view.setTag(R.id.dataBinding, this);
        synchronized (this) {
            this.x = 4L;
        }
        l();
    }

    @Override // androidx.databinding.ViewDataBinding
    public void c() {
        long j;
        synchronized (this) {
            j = this.x;
            this.x = 0L;
        }
        c.e.b.hf.e eVar = this.t;
        int i = ((j & 7) > 0L ? 1 : ((j & 7) == 0L ? 0 : -1));
        String str = null;
        if (i != 0) {
            String str2 = ((j & 6) == 0 || eVar == null) ? null : eVar.name;
            b.m.h hVar = eVar != null ? eVar.isCheck : null;
            ViewDataBinding.d dVar = ViewDataBinding.f259e;
            if (hVar == null) {
                ViewDataBinding.g gVar = this.k[0];
                if (gVar != null) {
                    gVar.a();
                }
            } else {
                ViewDataBinding.g[] gVarArr = this.k;
                ViewDataBinding.g gVar2 = gVarArr[0];
                if (gVar2 == null) {
                    k(0, hVar, dVar);
                } else if (gVar2.f268c != hVar) {
                    ViewDataBinding.g gVar3 = gVarArr[0];
                    if (gVar3 != null) {
                        gVar3.a();
                    }
                    k(0, hVar, dVar);
                }
            }
            boolean z = hVar != null ? hVar.f2335c : false;
            if (i != 0) {
                j |= z ? 16L : 8L;
            }
            r11 = z ? 0 : 8;
            str = str2;
        }
        if ((j & 6) != 0) {
            b.j.b.d.Q(this.v, str);
        }
        if ((j & 7) != 0) {
            this.w.setVisibility(r11);
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean e() {
        synchronized (this) {
            return this.x != 0;
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean i(int i, Object obj, int i2) {
        if (i != 0) {
            return false;
        }
        b.m.h hVar = (b.m.h) obj;
        if (i2 == 0) {
            synchronized (this) {
                this.x |= 1;
            }
            return true;
        }
        return false;
    }

    @Override // c.e.b.ff.g
    public void m(c.e.b.hf.e eVar) {
        this.t = eVar;
        synchronized (this) {
            this.x |= 2;
        }
        synchronized (this) {
            b.m.i iVar = this.f2328b;
            if (iVar != null) {
                iVar.b(this, 1, null);
            }
        }
        l();
    }
}