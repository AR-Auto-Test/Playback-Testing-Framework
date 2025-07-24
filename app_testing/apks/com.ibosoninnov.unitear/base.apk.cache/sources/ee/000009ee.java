package c.e.b.ef;

import android.view.LayoutInflater;
import android.view.ViewGroup;
import androidx.recyclerview.widget.RecyclerView;
import com.ibosoninnov.unitear.R;
import java.util.List;

/* compiled from: HistoryAdapter.java */
/* loaded from: classes2.dex */
public class d extends RecyclerView.g<a> {

    /* renamed from: a  reason: collision with root package name */
    public List<c.e.b.hf.e> f4717a;

    /* compiled from: HistoryAdapter.java */
    /* loaded from: classes2.dex */
    public static class a extends RecyclerView.d0 {

        /* renamed from: a  reason: collision with root package name */
        public final c.e.b.ff.g f4718a;

        public a(c.e.b.ff.g gVar) {
            super(gVar.s);
            this.f4718a = gVar;
        }
    }

    public d(List<c.e.b.hf.e> list) {
        this.f4717a = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.g
    public int getItemCount() {
        return this.f4717a.size();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [androidx.recyclerview.widget.RecyclerView$d0, int] */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public void onBindViewHolder(a aVar, int i) {
        a aVar2 = aVar;
        aVar2.f4718a.m(this.f4717a.get(i));
        aVar2.f4718a.r.setImageResource(this.f4717a.get(i).image);
    }

    /* JADX DEBUG: Return type fixed from 'androidx.recyclerview.widget.RecyclerView$d0' to match base method */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public a onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new a((c.e.b.ff.g) b.m.f.b(LayoutInflater.from(viewGroup.getContext()), R.layout.item_history, viewGroup, false));
    }
}