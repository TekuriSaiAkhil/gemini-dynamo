import React, {useState} from "react";
import axios from 'axios';
import Flashcard from './Flashcard.jsx';
import './Flashcard.css'

function App(){
  const [youtubeLink, setYoutubeLink] = useState("");
  const [keyConcept, setKeyConcept] = useState([]);

  const handleLinkChange = (event) => {
    setYoutubeLink(event.target.value);
  };

  const discardFlashcard = (index) => {
    setKeyConcept(currentConcepts => currentConcepts.filter((_, i) => i !== index));
  }

  const sendLink = async () => {
    try{
      const response = await axios.post("http://localhost:8000/analyze_video", {
        youtube_link: youtubeLink
      });
      
      const data = response.data;
      if(data.key_concepts && Array.isArray(data.key_concepts)){
        const transformedConcepts = data.key_concepts.map(concept => {
          const term = Object.keys(concept)[0];
          const definition = concept[term];
          return { term, definition }; 
        });
        setKeyConcept(transformedConcepts);
      }else{
        console.error("Data does not contain key concepts:", data)
        setKeyConcept([]);
      }
      console.log(keyConcept)
    }catch(error){
      console.log(error);
      setKeyConcept([]);
    }
  };

  return (
    <div className="App">
      <h1>Youtube link for Flashcards Generator</h1>
      <input
        type="text"
        placeholder="Paste Youtube Link Here"
        value={youtubeLink}
        onChange={handleLinkChange}
      />
      <button onClick={sendLink}>
        Generate Flashcards
      </button>
      <div className="flashcardsContainer">
        {keyConcept.map((concept, index) => (
          <Flashcard
            key = {index}
            term = {concept.term}
            definition = {concept.definition}
            onDiscard={()=>discardFlashcard(index)}
          />
        ))}
      </div>
    </div>
  )

}

export default App;