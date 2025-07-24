package c.e.b.ef;

import android.app.Activity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import androidx.recyclerview.widget.RecyclerView;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;

/* compiled from: ThumbnailAdapter.java */
/* loaded from: classes2.dex */
public class f extends RecyclerView.g<a> {

    /* renamed from: a  reason: collision with root package name */
    public ArrayList<c.e.b.hf.a> f4721a;

    /* renamed from: b  reason: collision with root package name */
    public Activity f4722b;

    /* renamed from: c  reason: collision with root package name */
    public c.e.b.gf.a f4723c;

    /* compiled from: ThumbnailAdapter.java */
    /* loaded from: classes2.dex */
    public static class a extends RecyclerView.d0 {

        /* renamed from: a  reason: collision with root package name */
        public ImageView f4724a;

        /* renamed from: b  reason: collision with root package name */
        public LinearLayout f4725b;

        /* renamed from: c  reason: collision with root package name */
        public ImageView f4726c;

        /* renamed from: d  reason: collision with root package name */
        public ProgressBar f4727d;

        public a(View view) {
            super(view);
            this.f4724a = (ImageView) view.findViewById(R.id.image);
            this.f4725b = (LinearLayout) view.findViewById(R.id.progressBarLayout);
            this.f4726c = (ImageView) view.findViewById(R.id.download);
            this.f4727d = (ProgressBar) view.findViewById(R.id.progressBar);
        }
    }

    public f(ArrayList<c.e.b.hf.a> arrayList, Activity activity) {
        this.f4721a = arrayList;
        this.f4722b = activity;
        this.f4723c = (c.e.b.gf.a) activity;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.g
    public int getItemCount() {
        return this.f4721a.size();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [androidx.recyclerview.widget.RecyclerView$d0, int] */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public void onBindViewHolder(a aVar, int i) {
        a aVar2 = aVar;
        aVar2.itemView.setOnClickListener(new e(this, i));
        c.c.a.b.d(this.f4722b).k(this.f4721a.get(i).thumbnail_url).j(R.drawable.image).f(R.drawable.image).B(aVar2.f4724a);
        if (this.f4721a.get(i).downloadStatus == -1) {
            aVar2.f4726c.setVisibility(0);
            aVar2.f4727d.setVisibility(4);
            aVar2.f4725b.setVisibility(4);
        } else if (this.f4721a.get(i).downloadStatus > -1 && this.f4721a.get(i).downloadStatus <= 100) {
            aVar2.f4726c.setVisibility(4);
            aVar2.f4727d.setVisibility(0);
            aVar2.f4725b.setVisibility(0);
            aVar2.f4727d.setProgress(this.f4721a.get(i).downloadStatus);
        } else if (this.f4721a.get(i).downloadStatus == 101) {
            aVar2.f4726c.setVisibility(4);
            aVar2.f4727d.setVisibility(4);
            aVar2.f4725b.setVisibility(4);
        }
    }

    /* JADX DEBUG: Return type fixed from 'androidx.recyclerview.widget.RecyclerView$d0' to match base method */
    @Override // androidx.recyclerview.widget.RecyclerView.g
    public a onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new a(LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_thumbnail, viewGroup, false));
    }
}