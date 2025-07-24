package c.e.b.ff;

import android.util.SparseIntArray;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.databinding.ViewDataBinding;
import com.ibosoninnov.unitear.R;

/* compiled from: ItemArGalleryBindingImpl.java */
/* loaded from: classes2.dex */
public class d extends c {
    public static final SparseIntArray C;
    public long D;

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        C = sparseIntArray;
        sparseIntArray.put(R.id.imageContainer, 1);
        sparseIntArray.put(R.id.imageContainer1, 2);
        sparseIntArray.put(R.id.imIcon0, 3);
        sparseIntArray.put(R.id.imIcon1, 4);
        sparseIntArray.put(R.id.imageContainer2, 5);
        sparseIntArray.put(R.id.imIcon2, 6);
        sparseIntArray.put(R.id.imIcon3, 7);
        sparseIntArray.put(R.id.labelCountTxt, 8);
        sparseIntArray.put(R.id.imIcon, 9);
        sparseIntArray.put(R.id.labelTxt, 10);
    }

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public d(b.m.e eVar, View view) {
        super(eVar, view, 0, (ImageView) r1[9], (ImageView) r1[3], (ImageView) r1[4], (ImageView) r1[6], (ImageView) r1[7], (LinearLayout) r1[1], (LinearLayout) r1[2], (LinearLayout) r1[5], (TextView) r1[8], (TextView) r1[10], (RelativeLayout) r1[0]);
        Object[] h2 = ViewDataBinding.h(eVar, view, 11, C);
        this.D = -1L;
        this.B.setTag(null);
        view.setTag(R.id.dataBinding, this);
        synchronized (this) {
            this.D = 2L;
        }
        l();
    }

    @Override // androidx.databinding.ViewDataBinding
    public void c() {
        synchronized (this) {
            this.D = 0L;
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean e() {
        synchronized (this) {
            return this.D != 0;
        }
    }

    @Override // androidx.databinding.ViewDataBinding
    public boolean i(int i, Object obj, int i2) {
        return false;
    }

    @Override // c.e.b.ff.c
    public void m(c.e.b.hf.b bVar) {
    }
}