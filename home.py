import TitlePage
import recommend
import quickRecommend
import TopMovies
from multiapp import MultiApp
import sys
app = MultiApp()
# st.set_page_config(page_title=’TrekViz’, page_icon=”🖖”)
app.add_app("Home Page", TitlePage.app)
app.add_app("PinPoint Recommendation", recommend.app)
app.add_app("Quick Recommendation", quickRecommend.app)
app.add_app("Top Movies", TopMovies.app)
app.run()
