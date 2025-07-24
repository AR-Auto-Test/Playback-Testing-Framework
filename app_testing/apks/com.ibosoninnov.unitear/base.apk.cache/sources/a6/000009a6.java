package c.e.b;

import android.media.MediaPlayer;
import java.util.Objects;
import java.util.function.BiConsumer;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class ba implements BiConsumer {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ String f4575a;

    /* JADX DEBUG: Marked for inline */
    /* JADX DEBUG: Method not inlined, still used in: [c.e.b.r.run():void, com.ibosoninnov.unitear.ARCoreSceneformActivity.onResume():void] */
    public /* synthetic */ ba(String str) {
        this.f4575a = str;
    }

    @Override // java.util.function.BiConsumer
    public final void accept(Object obj, Object obj2) {
        String str = this.f4575a;
        MediaPlayer mediaPlayer = (MediaPlayer) obj;
        if (((Boolean) obj2).booleanValue() && Objects.equals(str, "reload")) {
            mediaPlayer.seekTo(0);
        }
        mediaPlayer.start();
    }
}