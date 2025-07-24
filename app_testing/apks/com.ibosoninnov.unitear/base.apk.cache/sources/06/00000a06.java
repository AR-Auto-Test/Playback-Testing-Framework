package c.e.b.ff;

import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.databinding.ViewDataBinding;
import com.ibosoninnov.unitear.R;

/* compiled from: ItemAboutBindingImpl.java */
/* loaded from: classes2.dex */
public class b extends a {
    public final TextView s;
    public final TextView t;
    public long u;

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public b(b.m.e eVar, View view) {
        super(eVar, view, 0, (LinearLayout) r0[0]);
        Object[] h2 = ViewDataBinding.h(eVar, view, 3, null);
        this.u = -1L;
        this.r.setTag(null);
        TextView textView = (TextView) h2[1];
        this.s = textView;
        textView.setTag(null);
        TextView textView2 = (TextView) h2[2];
        this.t = textView2;
        textView2.setTag(null);
        view.setTag(R.id.dataBinding, this);
        synchronized (this) {
            this.u = 2L;
        }
        l();
    }

    @Override // androidx.databinding.ViewDataBinding
    public void c() {
        long j;
        synchronized (this) {
            j = this.u;
            this.u = 0L;
        }
        if ((j & 3) != 0) {
            b.j.b.d.Q(this.s, null);
            b.j.b.d.Q(this.t, null);
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean e() {
        synchronized (this) {
            return this.u != 0;
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean i(int i, Object obj, int i2) {
        return false;
    }
}